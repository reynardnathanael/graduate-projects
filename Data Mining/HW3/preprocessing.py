import os
import re
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from string import punctuation
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import DistilBertModel, DistilBertTokenizerFast, BertModel, BertTokenizerFast, RobertaModel, RobertaTokenizerFast

# Constants variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'distilbert-base-uncased'
# MODEL_NAME = 'bert-base-uncased'
# MODEL_NAME = 'roberta-base'
MAX_LEN = 128
BATCH_SIZE = 16
DATA_DIR = './data'

# Seeding for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Clean text from unwanted characters
def clean_text(text):
    if pd.isna(text):
        return text
    
    # Regex patterns
    p_marks = '[' + re.escape(punctuation + '-') + ']'
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\S+|#\S+', '', text)
    text = re.sub(r'\b(\w+)-(\1)\b', r'\1', text)
    text = re.sub(p_marks, ' ', text)
    text = re.sub(r'[^a-zA-Z\s]+', '', text)
    text = text.replace('\n', ' ')
    text = text.strip(' ')
    return text.lower()

# Filling the missing values (NaNs)
def impute_missing_values(news_data):
    imputation_log = []
    for feature in news_data.columns:
        null_count = news_data[feature].isna().sum()
        if null_count > 0:
            imputation_log.append({feature: str(null_count)})
            if feature in ['title_entities', 'abstract_entities']:
                fill_value = '""'
            else:
                fill_value = f'[BLANK_{feature}]'
            news_data[feature] = news_data[feature].fillna(fill_value)
    print(f'Filled columns: {imputation_log}')
    return news_data

# Encoder class for news titles using BERT family models
class NewsEncoder:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(device)
        self.device = device

    def encode_batch(self, texts):
        inputs = self.tokenizer(
            texts, 
            return_tensors='pt', 
            max_length=MAX_LEN, 
            truncation=True, 
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return cls_embeddings

# Generate the news embeddings
def generate_news_embeddings(text_list, cache_name="news_embeddings_title_abstract.npy"):
    cache_path = os.path.join(DATA_DIR, cache_name)

    if os.path.exists(cache_path):
        return np.load(cache_path)

    encoder = NewsEncoder()
    embeddings = []
    total_batches = (len(text_list) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(text_list), BATCH_SIZE), total=total_batches, desc="Encoding"):
        batch_texts = text_list[i : i + BATCH_SIZE]
        batch_emb = encoder.encode_batch(batch_texts)
        embeddings.append(batch_emb)

    full_embeddings = np.vstack(embeddings)

    np.save(cache_path, full_embeddings)
    return full_embeddings

# User Embedding Construction
def build_user_embeddings(df_behaviors, news_emb_map, vector_dim=768):
    user_embeddings = {}
    
    for row in tqdm(df_behaviors.itertuples(index=False), total=len(df_behaviors)):
        user_id = row.user_id
        clicked_ids = str(row.clicked_news).split()
        
        # Sum vectors
        user_vec = np.zeros(vector_dim, dtype=np.float32)
        count = 0
        
        for nid in clicked_ids:
            if nid in news_emb_map:
                user_vec += news_emb_map[nid]
                count += 1
        
        # Average
        if count > 0:
            user_vec /= count
            
        user_embeddings[user_id] = user_vec
        
    return user_embeddings

# Create training/testing pairs
def create_pairs(df_behaviors, user_emb_map, news_emb_map, is_train=True, vector_dim=768):
    X_data = []
    Y_data = []
    Imp_IDs = []
    
    print(f"Creating {'Training' if is_train else 'Testing'} Pairs...")
    
    for row in tqdm(df_behaviors.itertuples(index=False), total=len(df_behaviors)):
        u_vec = user_emb_map.get(row.user_id, np.zeros(vector_dim, dtype=np.float32))
        impressions = str(row.impressions).split()
        
        for imp in impressions:
            parts = imp.split('-')
            
            if is_train:
                if len(parts) != 2: continue
                news_id, label = parts
                label = int(label)
                
                n_vec = news_emb_map.get(news_id, np.zeros(vector_dim, dtype=np.float32))
                X_data.append((u_vec, n_vec))
                Y_data.append(label)
                
            else:
                news_id = parts[0]
                n_vec = news_emb_map.get(news_id, np.zeros(vector_dim, dtype=np.float32))
                
                X_data.append((u_vec, n_vec))
                Imp_IDs.append(row.id)
                
    return X_data, Y_data, Imp_IDs