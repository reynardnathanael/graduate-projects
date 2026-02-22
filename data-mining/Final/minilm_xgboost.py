import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

# Load the datasets
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['response'] = np.nan 

full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Extract Complexity Features (Mathematical Complexity)
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

# Embeddings using MiniLM (SentenceTransformer)
full_df['pure_text'] = (
    "Category: " + full_df['category'].astype(str) + 
    ". Concepts: " + full_df['concepts_of_the_problem'].astype(str)
)
bert = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = bert.encode(full_df['pure_text'].tolist(), show_progress_bar=True)

# PCA for dimensionality reduction
pca = PCA(n_components=20, random_state=42)
pca_features = pca.fit_transform(embeddings)
pca_cols = [f'pca_{i}' for i in range(20)]
pca_df = pd.DataFrame(pca_features, columns=pca_cols)
full_df = pd.concat([full_df, pca_df], axis=1)

# Semantic Clustering
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
full_df['cluster_id'] = kmeans.fit_predict(embeddings)

# Target Encoding
full_df['feat_cluster_diff'] = np.nan
full_df['feat_student_global'] = np.nan
full_df['feat_specialist'] = np.nan
full_df['is_imputed'] = 0
global_mean = train_df['response'].mean()

# The first K-fold for target encoding
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
train_indices = full_df[full_df['is_train'] == 1].index

# Target Encoding (Train)
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

# Impute any remaining NaNs with global mean
full_df.fillna(global_mean, inplace=True)

# Additional Feature: Problem Length
full_df['problem_len'] = full_df['Problem'].apply(str).apply(len)

# All the features to be used for modeling
features = [
    'comp_latex', 'comp_nums', 'comp_len',
    'has_frac', 'has_sqrt', 'has_dec', 'has_perc',
    'feat_cluster_diff', 'feat_student_global', 'feat_specialist', 'is_imputed',
    'cluster_id'
] + pca_cols

train_data = full_df[full_df['is_train'] == 1].copy()
test_data = full_df[full_df['is_train'] == 0].copy()

X = train_data[features].to_numpy()
y = train_data['response'].to_numpy()
X_test = test_data[features].to_numpy()

# Scaler for SVM and Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Second K-fold for stacking ensemble
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# OOF Arrays
oof_xgb = np.zeros(len(X))
oof_svm = np.zeros(len(X))
oof_lr  = np.zeros(len(X))

test_xgb = np.zeros(len(X_test))
test_svm = np.zeros(len(X_test))
test_lr  = np.zeros(len(X_test))

# First Model: XGBoost
model_xgb = XGBClassifier(
    n_estimators=1000, learning_rate=0.02, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    enable_categorical=True, random_state=42, n_jobs=-1
)

# Second Model: Calibrated SVM
svm_base = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
model_svm = CalibratedClassifierCV(svm_base, method='sigmoid', cv=3)

# Third Model: Simple Logistic Regression
model_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

# Training Base Models and Generating OOF Predictions
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    X_tr_s, X_val_s = X_scaled[tr_idx], X_scaled[val_idx]
    
    # XGBoost
    model_xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / 10
    
    # SVM
    model_svm.fit(X_tr_s, y_tr)
    oof_svm[val_idx] = model_svm.predict_proba(X_val_s)[:, 1]
    test_svm += model_svm.predict_proba(X_test_scaled)[:, 1] / 10
    
    # Logistic Regression
    model_lr.fit(X_tr_s, y_tr)
    oof_lr[val_idx] = model_lr.predict_proba(X_val_s)[:, 1]
    test_lr += model_lr.predict_proba(X_test_scaled)[:, 1] / 10
    
    print(f"Fold {fold+1} complete")

# Stacking Ensemble
stack_train = pd.DataFrame({'xgb': oof_xgb, 'svm': oof_svm, 'lr': oof_lr})
stack_test = pd.DataFrame({'xgb': test_xgb, 'svm': test_svm, 'lr': test_lr})

# Meta-Model: Logistic Regression
meta_model = LogisticRegression(fit_intercept=True)
meta_model.fit(stack_train, y)

print("Meta-Learner Weights:")
print(f"XGB: {meta_model.coef_[0][0]:.4f}")
print(f"SVM: {meta_model.coef_[0][1]:.4f}")
print(f"LR:  {meta_model.coef_[0][2]:.4f}")

# Meta-Evaluation
oof_final = meta_model.predict_proba(stack_train)[:, 1]
score = roc_auc_score(y, oof_final)
print(f"\nStacked OOF AUC: {score:.4f}")

# Cross-validated Stacked AUC
fold_scores = []
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    X_meta_val = stack_train.iloc[val_idx]
    y_meta_val = y[val_idx]
    
    # Predict using the meta-model
    val_preds = meta_model.predict_proba(X_meta_val)[:, 1]
    
    score = roc_auc_score(y_meta_val, val_preds)
    fold_scores.append(score)
    print(f"Fold {fold+1} Stacked AUC: {score:.4f}")

print(f"\nMean Stacked AUC: {np.mean(fold_scores):.4f}")
print(f"Std Dev: {np.std(fold_scores):.4f}")

# Final Test Predictions
final_preds = meta_model.predict_proba(stack_test)[:, 1]

# Generate Submission
submission = pd.DataFrame({
    'interaction_id': test_df['interaction_id'],
    'response': final_preds
})

submission.to_csv('submission.csv', index=False)
print("Saved submission.csv")