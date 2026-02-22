import json
import os
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_DIR = "./dataset"
INPUT_MERGE_FILE = "merge.json"       
OUTPUT_MERGE_FILE = "merge_hybrid.json" 
MODEL_NAME = "sujet-ai/Marsilia-Embeddings-EN-Base"
TOP_K_NEW = 50 
BATCH_SIZE = 128 

# List of all subfolders/tasks to process
TASKS = [
    "FinanceBench", 
    "FinQABench", 
    "FinDER", 
    "ConvFinQA", 
    "FinQA", 
    "TATQA", 
    "MultiHiertt"
]

def load_jsonl(filepath):
    """Loads a JSONL file and returns IDs and Data list."""
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: {filepath} not found. Skipping...")
        return [], []
        
    data = []
    ids = []
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
            ids.append(item['_id'])
    return ids, data

def main():
    print(f"--- Starting Hybrid Retrieval (Per Task) ---")
    
    print(f"Loading Model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    
    for task in TASKS:
        print(f"\n🔹 Processing Task: {task}")
        task_dir = os.path.join(DATASET_DIR, task)
        
        # Paths for this specific task
        corpus_path = os.path.join(task_dir, "corpus.jsonl")
        query_path = os.path.join(task_dir, "queries.jsonl")
        merge_path = os.path.join(task_dir, INPUT_MERGE_FILE) # Look inside the task folder

        # 1. Load Merge File (Existing Candidates)
        if not os.path.exists(merge_path):
            print(f"⚠️ {INPUT_MERGE_FILE} not found in {task_dir}. Skipping task.")
            continue
            
        print(f"   - Loading existing merge file: {merge_path}...")
        with open(merge_path, 'r') as f:
            task_merge_data = json.load(f)

        # 2. Load Corpus and Queries
        corpus_ids, corpus_data = load_jsonl(corpus_path)
        query_ids, query_data = load_jsonl(query_path)

        if not corpus_ids or not query_ids:
            print(f"Skipping {task} due to missing corpus/queries.")
            continue

        # 3. Embed (GPU)
        corpus_texts = [f"{doc.get('title', '')} {doc.get('text', '')}".strip() for doc in corpus_data]
        query_texts = [q['text'] for q in query_data]

        print(f"   - Encoding {len(corpus_texts)} Documents...")
        corpus_embeddings = model.encode(corpus_texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_tensor=True)
        
        print(f"   - Encoding {len(query_texts)} Queries...")
        query_embeddings = model.encode(query_texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_tensor=True)

        # 4. Search
        print(f"   - Running Vector Search...")
        search_results = util.semantic_search(query_embeddings, corpus_embeddings, top_k=TOP_K_NEW)

        # 5. Merge Results
        print(f"   - Merging results...")
        for i, q_id in enumerate(query_ids):
            # Get existing BM25 candidates
            current_list = task_merge_data.get(q_id, [])
            candidate_set = set(current_list)

            # Add new Marsilia Vector candidates
            hits = search_results[i]
            for hit in hits:
                corpus_idx = hit['corpus_id']
                real_corpus_id = corpus_ids[corpus_idx]
                candidate_set.add(real_corpus_id)
            
            # Save back to dict
            task_merge_data[q_id] = list(candidate_set)

        # 6. Save Hybrid File LOCALLY to the task folder
        output_path = os.path.join(task_dir, OUTPUT_MERGE_FILE)
        print(f"💾 Saving Hybrid Dataset to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(task_merge_data, f)
        
    print("\n✅ Done! Hybrid merge complete for all tasks.")

if __name__ == "__main__":
    main()