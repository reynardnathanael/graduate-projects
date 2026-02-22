import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. SETUP & DATA LOADING
# ---------------------------------------------------------
print("Initializing Environment...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['response'] = np.nan 

full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING (The Winning Set)
# ---------------------------------------------------------
print("Step 1/4: Feature Engineering...")

def extract_complexity(text):
    text = str(text).lower()
    latex_count = text.count('\\')
    nums = re.findall(r'\d+', text)
    num_count = len(nums)
    has_fraction = 1 if 'frac' in text or '/' in text else 0
    has_sqrt = 1 if 'sqrt' in text else 0
    has_decimal = 1 if '.' in text and re.search(r'\d\.\d', text) else 0
    has_percent = 1 if '%' in text else 0
    length = len(text)
    return pd.Series([latex_count, num_count, has_fraction, has_sqrt, has_decimal, has_percent, length])

complexity_cols = ['comp_latex', 'comp_nums', 'has_frac', 'has_sqrt', 'has_dec', 'has_perc', 'comp_len']
full_df[complexity_cols] = full_df['Problem'].apply(extract_complexity)

# Embeddings (MiniLM)
full_df['pure_text'] = (
    "Category: " + full_df['category'].astype(str) + 
    ". Concepts: " + full_df['concepts_of_the_problem'].astype(str)
)
bert = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = bert.encode(full_df['pure_text'].tolist(), show_progress_bar=True)

# PCA
pca = PCA(n_components=20, random_state=42)
pca_features = pca.fit_transform(embeddings)
pca_cols = [f'pca_{i}' for i in range(20)]
pca_df = pd.DataFrame(pca_features, columns=pca_cols)
full_df = pd.concat([full_df, pca_df], axis=1)

# Clustering
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
full_df['cluster_id'] = kmeans.fit_predict(embeddings)

# Target Encoding
full_df['feat_cluster_diff'] = np.nan
full_df['feat_student_global'] = np.nan
full_df['feat_specialist'] = np.nan
full_df['is_imputed'] = 0
global_mean = train_df['response'].mean()

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
train_indices = full_df[full_df['is_train'] == 1].index

for tr_idx, val_idx in kf.split(train_indices, full_df.loc[train_indices, 'response']):
    global_tr = train_indices[tr_idx]
    global_val = train_indices[val_idx]
    X_tr = full_df.loc[global_tr]
    
    cluster_diff = X_tr.groupby('cluster_id')['response'].mean()
    student_global = X_tr.groupby('student_id')['response'].mean()
    specialist = X_tr.groupby(['student_id', 'cluster_id'])['response'].mean()
    
    full_df.loc[global_val, 'feat_cluster_diff'] = full_df.loc[global_val, 'cluster_id'].map(cluster_diff)
    full_df.loc[global_val, 'feat_student_global'] = full_df.loc[global_val, 'student_id'].map(student_global)
    
    val_subset = full_df.loc[global_val, ['student_id', 'cluster_id']].copy()
    val_subset['response'] = val_subset.set_index(['student_id', 'cluster_id']).index.map(specialist)
    full_df.loc[global_val, 'feat_specialist'] = val_subset['response']
    full_df.loc[global_val[val_subset['response'].isna()], 'is_imputed'] = 1

# Test Encoding
all_train = full_df[full_df['is_train'] == 1]
cluster_diff_all = all_train.groupby('cluster_id')['response'].mean()
student_global_all = all_train.groupby('student_id')['response'].mean()
specialist_all = all_train.groupby(['student_id', 'cluster_id'])['response'].mean()

test_indices = full_df[full_df['is_train'] == 0].index
full_df.loc[test_indices, 'feat_cluster_diff'] = full_df.loc[test_indices, 'cluster_id'].map(cluster_diff_all)
full_df.loc[test_indices, 'feat_student_global'] = full_df.loc[test_indices, 'student_id'].map(student_global_all)
test_subset = full_df.loc[test_indices, ['student_id', 'cluster_id']].copy()
test_subset['response'] = test_subset.set_index(['student_id', 'cluster_id']).index.map(specialist_all)
full_df.loc[test_indices, 'feat_specialist'] = test_subset['response']
full_df.loc[test_indices[test_subset['response'].isna()], 'is_imputed'] = 1

# WINNING IMPUTATION
full_df.fillna(global_mean, inplace=True)

# ---------------------------------------------------------
# 3. PREPARE DATA FOR TWO-TOWER
# ---------------------------------------------------------
print("Step 2/4: Preparing Data for Two-Tower...")

full_df['problem_len'] = full_df['Problem'].apply(str).apply(len)

# Define Tower Inputs
# Tower A Features (User/Skill)
user_features = [
    'feat_cluster_diff', 'feat_student_global', 'feat_specialist', 'is_imputed'
]

# Tower B Features (Question/Content)
question_features = [
    'comp_latex', 'comp_nums', 'comp_len',
    'has_frac', 'has_sqrt', 'has_dec', 'has_perc',
    'cluster_id'
] + pca_cols

train_data = full_df[full_df['is_train'] == 1].copy()
test_data = full_df[full_df['is_train'] == 0].copy()

# Split into two numpy arrays per dataset
X_user_train = train_data[user_features].to_numpy()
X_question_train = train_data[question_features].to_numpy()
y_train = train_data['response'].to_numpy()

X_user_test = test_data[user_features].to_numpy()
X_question_test = test_data[question_features].to_numpy()

# Scale
scaler_user = StandardScaler()
X_user_train = scaler_user.fit_transform(X_user_train)
X_user_test = scaler_user.transform(X_user_test)

scaler_question = StandardScaler()
X_question_train = scaler_question.fit_transform(X_question_train)
X_question_test = scaler_question.transform(X_question_test)

# ---------------------------------------------------------
# 4. TWO-TOWER MODEL DEFINITION
# ---------------------------------------------------------
class HybridTwoTower(nn.Module):
    def __init__(self, skill_dim, bert_dim):
        super(HybridTwoTower, self).__init__()
        
        # Tower A: Skill Processing
        self.user_tower = nn.Sequential(
            nn.Linear(skill_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4), # High Dropout for small data
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Tower B: Question Processing
        self.question_tower = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4), # High Dropout for small data
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 64), # 32 + 32
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, skill_vec, question_vec):
        u_out = self.user_tower(skill_vec)
        q_out = self.question_tower(question_vec)
        combined = torch.cat([u_out, q_out], dim=1)
        return self.classifier(combined)

# ---------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------
print("Step 3/4: Training Two-Tower...")

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []
test_preds = np.zeros(len(test_data))

skill_dim = X_user_train.shape[1]
bert_dim = X_question_train.shape[1]

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_user_train, y_train)):
    # Slice Data
    u_tr = torch.tensor(X_user_train[tr_idx], dtype=torch.float32).to(device)
    q_tr = torch.tensor(X_question_train[tr_idx], dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train[tr_idx], dtype=torch.float32).unsqueeze(1).to(device)
    
    u_val = torch.tensor(X_user_train[val_idx], dtype=torch.float32).to(device)
    q_val = torch.tensor(X_question_train[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.float32).unsqueeze(1).to(device)
    
    model = HybridTwoTower(skill_dim, bert_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_auc = 0
    patience = 20
    counter = 0
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(u_tr, q_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_out = model(u_val, q_val)
            auc = roc_auc_score(y_val.cpu(), val_out.cpu())
            
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
            counter = 0
            best_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                break
    
    model.load_state_dict(best_state)
    cv_scores.append(best_auc)
    print(f"Fold {fold+1} AUC: {best_auc:.4f}")
    
    # Predict Test
    model.eval()
    with torch.no_grad():
        u_test_ts = torch.tensor(X_user_test, dtype=torch.float32).to(device)
        q_test_ts = torch.tensor(X_question_test, dtype=torch.float32).to(device)
        test_preds += model(u_test_ts, q_test_ts).cpu().numpy().flatten() / 10

print(f"\nAverage CV AUC (Two-Tower): {np.mean(cv_scores):.4f}")

# ---------------------------------------------------------
# 6. SUBMISSION
# ---------------------------------------------------------
submission = pd.DataFrame({
    'interaction_id': test_df['interaction_id'],
    'response': test_preds
})

submission.to_csv('submission_twotower_minilm.csv', index=False)
print("Saved submission_twotower_minilm.csv")