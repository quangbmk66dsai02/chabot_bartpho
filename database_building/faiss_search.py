
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

import argparse
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import pandas as pd
# Add chatbot_qa directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from utils
from utils.common_utils import get_absolute_path


def load_faiss_index_and_mapping(faiss_index_path, mapping_csv_path):
    """
    Loads the FAISS index and the mapping DataFrame.

    Parameters:
    - faiss_index_path (str): Path to the saved FAISS index.
    - mapping_csv_path (str): Path to the saved mapping CSV.

    Returns:
    - faiss.Index: The loaded FAISS index.
    - pd.DataFrame: The loaded mapping DataFrame.
    """
    # Load FAISS index
    index = faiss.read_index(faiss_index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors.")

    # Load mapping DataFrame
    mapping_df = pd.read_csv(mapping_csv_path, encoding='utf-8-sig')
    print(f"Loaded mapping DataFrame with {len(mapping_df)} entries.")

    return index, mapping_df

def get_query_embedding(model, tokenizer, query, device, max_length=256):
    """
    Generates an embedding for a single query using the fine-tuned PhoBERT model.

    Parameters:
    - model (transformers.AutoModel): The fine-tuned PhoBERT model.
    - tokenizer (transformers.AutoTokenizer): The PhoBERT tokenizer.
    - query (str): The input query string.
    - device (torch.device): The device to run the model on.
    - max_length (int): Maximum token length for the query.

    Returns:
    - np.ndarray: The embedding vector.
    """
    model.eval()
    with torch.no_grad():
        encoding = tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding='max_length',  # Padding handled in DataLoader or here
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # Shape: (1, seq_length, hidden_size)

        # Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask  # Shape: (1, hidden_size)

        # Convert to NumPy array
        embedding = embedding.cpu().numpy().astype('float32')

    return embedding

def search_faiss(query, model, tokenizer, index, mapping_df, device, k=5):
    """
    Performs a similarity search for the given query using FAISS.

    Parameters:
    - query (str): The search query string.
    - model (transformers.AutoModel): The fine-tuned PhoBERT model.
    - tokenizer (transformers.AutoTokenizer): The PhoBERT tokenizer.
    - index (faiss.Index): The FAISS index.
    - mapping_df (pd.DataFrame): The mapping DataFrame.
    - device (torch.device): The device to run the model on.
    - k (int): Number of nearest neighbors to retrieve.

    Returns:
    - List[Dict]: A list of search results with metadata.
    """
    # Generate embedding for the query
    query_embedding = get_query_embedding(model, tokenizer, query, device)

    # Perform the search
    distances, indices = index.search(query_embedding, k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx < len(mapping_df):
            result = {
                'rank': rank,
                'question': mapping_df.iloc[idx]['question'],
                'answer_chunk': mapping_df.iloc[idx]['answer_chunk'],
                'original_index': mapping_df.iloc[idx]['original_index'],
                'distance': distances[0][rank-1]
            }
            results.append(result)
        else:
            # Handle out-of-bounds indices
            print(f"Index {idx} is out of bounds for the mapping DataFrame.")

    return results

def main():
    # Argument parsing for flexibility
    parser = argparse.ArgumentParser(description="FAISS Search with Fine-Tuned PhoBERT")
    parser.add_argument('--query', type=str, default="Q: Những bài học gì có thể rút ra từ chiến thắng Bạch Đằng lần thứ hai năm 1288 cho các thế hệ sau?", help='Search query string')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to retrieve')
    parser.add_argument('--faiss_index', type=str, default=get_absolute_path('../database_building/faiss_index.bin'), help='Path to FAISS index file')
    parser.add_argument('--mapping_csv', type=str, default=get_absolute_path('../database_building/faiss_mapping.csv'), help='Path to mapping CSV file')
    parser.add_argument('--model_path', type=str, default=get_absolute_path('../pho_embedding_train/fine-tuned-phobert-embedding-model'), help='Path to fine-tuned PhoBERT model')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the fine-tuned PhoBERT model and tokenizer
    print("Loading the fine-tuned PhoBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    print("Model and tokenizer loaded successfully.")

    # Load the FAISS index and mapping
    print("Loading FAISS index and mapping DataFrame...")
    index, mapping_df = load_faiss_index_and_mapping(args.faiss_index, args.mapping_csv)
    print('THIS IS TYPE OF ID', type(index))

    # Perform the search
    query = args.query
    k = args.k

    print(f"\nPerforming search for query: '{query}' with top {k} results...\n")
    search_results = search_faiss(query, model, tokenizer, index, mapping_df, device, k=k)

    # Display the results
    for res in search_results:
        print(f"--- Rank {res['rank']} ---")
        print(f"Question: {res['question']}")
        print(f"Answer Chunk: {res['answer_chunk']}")
        print(f"Original QA Pair Index: {res['original_index']}")
        print(f"Distance: {res['distance']:.4f}")
        print("-------------------------\n")

if __name__ == "__main__":
    main()
