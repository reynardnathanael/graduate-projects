import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Import functions and classes from other modules
from model import TwoTowerModel
from preprocessing import set_seed, clean_text, impute_missing_values, generate_news_embeddings, build_user_embeddings, create_pairs, DEVICE, DATA_DIR
from training import NewsDataset, Trainer, generate_submission

# Constant variables for pre-processing
SEED = 42
OUTPUT_FILE = 'dataset_processed_fulltext.pt'

# Constant variables for training and evaluation
BATCH_SIZE = 32 
INPUT_DIM = 1536

def main():
    # START PRE-PROCESSING

    set_seed(SEED)
    
    # Load datasets
    train_behaviors = pd.read_csv(f"{DATA_DIR}/train/train_behaviors.tsv", sep="\t", low_memory=False).iloc[:-1]
    test_behaviors = pd.read_csv(f"{DATA_DIR}/test/test_behaviors.tsv", sep="\t", low_memory=False)
    
    train_news = pd.read_csv(f"{DATA_DIR}/train/train_news.tsv", sep="\t", low_memory=False).iloc[:-1]
    test_news = pd.read_csv(f"{DATA_DIR}/test/test_news.tsv", sep="\t", low_memory=False)

    # Load entity embeddings
    # train_entity_path = f"{DATA_DIR}/train/train_entity_embedding.vec"
    # test_entity_path = f"{DATA_DIR}/test/test_entity_embedding.vec"
    
    # entity_emb_dict = load_entity_embeddings(train_entity_path)
    # test_entities = load_entity_embeddings(test_entity_path)
    # entity_emb_dict.update(test_entities)

    # entity_emb_list = []
    # for idx, row in tqdm(all_news.iterrows(), total=len(all_news)):
    #     entity_vec = extract_news_entity_embedding(
    #         row['title_entities'], 
    #         row['abstract_entities'], 
    #         entity_emb_dict,
    #         entity_dim=100
    #     )
    #     entity_emb_list.append(entity_vec)
    
    # entity_emb_matrix = np.array(entity_emb_list)
    # text_emb_normalized = normalize(text_emb_matrix, axis=1, norm='l2')
    # entity_emb_normalized = normalize(entity_emb_matrix, axis=1, norm='l2')
    
    # Combine news to get unique set
    all_news = pd.concat([train_news, test_news]).drop_duplicates("news_id").reset_index(drop=True)
    
    # Fill NaNs
    all_news = impute_missing_values(all_news)

    # Cleaning
    all_news['title'] = all_news['title'].apply(clean_text)
    all_news['abstract'] = all_news['abstract'].apply(clean_text)
    
    # Combine title and abstract
    all_news['full_text'] = all_news['title'] + " " + all_news['abstract']

    # Generate (or load) news embeddings
    emb_matrix = generate_news_embeddings(
        all_news['full_text'].tolist(), 
        cache_name="news_embeddings_title_abstract.npy"
    )

    # combined_emb_matrix = np.concatenate([emb_matrix, entity_emb_scaled], axis=1)
    
    # Create news_id to embedding map
    news_emb_map = dict(zip(all_news['news_id'], emb_matrix))
    
    # Create user embeddings
    train_user_map = build_user_embeddings(train_behaviors, news_emb_map)
    test_user_map = build_user_embeddings(test_behaviors, news_emb_map)
    
    # Prepare final lists
    x_train_full, y_train_full, _ = create_pairs(train_behaviors, train_user_map, news_emb_map, is_train=True)
    x_test, _, imp_ids_test = create_pairs(test_behaviors, test_user_map, news_emb_map, is_train=False)
    
    # Split training data
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, 
        y_train_full, 
        test_size=0.1, 
        random_state=SEED
    )
    
    # Save processed data
    torch.save({
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'impressions_id': imp_ids_test
    }, OUTPUT_FILE)

    # END PRE-PROCESSING

    # START TRAINING AND EVALUATION

    set_seed(SEED)

    # Load Data
    data = torch.load(OUTPUT_FILE, weights_only=False)
    
    # Create Datasets
    train_ds = NewsDataset(data['x_train'], data['y_train'])
    val_ds = NewsDataset(data['x_val'], data['y_val'])
    test_ds = NewsDataset(data['x_test'], y_data=None)
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    model = TwoTowerModel(input_dim=INPUT_DIM)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    trained_model = trainer.fit()
    trainer.plot_history()
    
    # Save Model
    torch.save(trained_model, "two_tower_model_final.pth")
    
    # Generate Submission
    generate_submission(
        trained_model, 
        test_loader, 
        data['impressions_id'], 
        DEVICE
    )

    # END TRAINING AND EVALUATION

if __name__ == "__main__":
    main()