import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import pandas as pd
import numpy as np
import random
import json
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
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask

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
    query_embedding = get_query_embedding(model, tokenizer, query, device)
    distances, indices = index.search(query_embedding, k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx < len(mapping_df):
            result = {
                'rank': rank,
                'question': mapping_df.iloc[idx]['question'],
                'answer_chunk': mapping_df.iloc[idx]['answer_chunk'],
                'original_index': mapping_df.iloc[idx]['original_index'],
                'distance': distances[0][rank - 1]
            }
            results.append(result)
        else:
            print(f"Index {idx} is out of bounds for the mapping DataFrame.")

    return results

def rerank_results_bge(query, results, device='cuda', num=1):
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel('BAAI/bge-m3',
                        use_fp16=True,
                        pooling_method='cls',
                        devices=device) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    corpus =  results
    query = [query]
    embeddings_1 = model.encode_queries(
        query,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    embeddings_2 = model.encode_corpus(
        corpus=corpus,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_similarity = embeddings_1["dense_vecs"] @ embeddings_2["dense_vecs"].T
    sparse_similarity = model.compute_lexical_matching_score(
        embeddings_1["lexical_weights"],
        embeddings_2["lexical_weights"],
    )

    sparse_similarity = np.array(sparse_similarity)
    sparse_similarity = sparse_similarity[0]
    max2_indices = np.argsort(sparse_similarity)[::-1].tolist()
    new_max2_indices = max2_indices[:num]

    dense_similarity = np.array(dense_similarity)
    dense_similarity = dense_similarity[0]
    max_indices = np.argsort(dense_similarity)[::-1].tolist()
    new_max_indices = max_indices[:num]


    reranked_results = []
    for i in range(num):
        id = new_max_indices[i]
        id2 = new_max2_indices[i]
        reranked_results.append({'sparse_score': sparse_similarity[id],'dense_score': dense_similarity[id], 'answer_chunk': results[id]})
        reranked_results.append({'sparse_score': sparse_similarity[id2],'dense_score': dense_similarity[id2], 'answer_chunk': results[id2]})
    unique_reranked_results = []
    seen = set()

    for d in reranked_results:
        # Convert dictionary to a tuple of its items (hashable)
        dict_tuple = tuple(d.items())
        if dict_tuple not in seen:
            seen.add(dict_tuple)
            unique_reranked_results.append(d)
    print(len(unique_reranked_results), len(reranked_results))
    return unique_reranked_results

def main():
    parser = argparse.ArgumentParser(description="FAISS Search with Fine-Tuned PhoBERT and Reranker")
    # parser.add_argument('--query', type=str, default= tmp_q, help='Search query string')
    parser.add_argument('--k', type=int, default=100, help='Number of nearest neighbors to retrieve')
    # parser.add_argument('--faiss_index', type=str, default=get_absolute_path('../database_building/test_data/faiss_index_test.bin'), help='Path to FAISS index file')
    # parser.add_argument('--mapping_csv', type=str, default=get_absolute_path('../database_building/test_data/faiss_mapping_test.csv'), help='Path to mapping CSV file')
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
    k = 100
    num = 10

    #Randomly select quest
    json_file_path = "api_QA/json_data/parsed_qa_pairs_output_mixed_6k.json"  # Update the path as needed
    # Load the JSON data
    with open(json_file_path, "r", encoding="utf-8") as file:
        qa_pairs = json.load(file)
    df = pd.DataFrame(qa_pairs)
    total_questtion_list = []
    for tmp_id, row in df.iterrows():
        question = row['question']
        total_questtion_list.append(question)
    start = 1
    print("total quest number", len(total_questtion_list))
    end = len(total_questtion_list) - 1
    indices_reranked_passed_list = []
    indices_reranked_total_list = []
    # Generate a list of 100 unique random numbers
    random_numbers = random.sample(range(start, end+1), 100)
    for random_id in random_numbers:
        query = total_questtion_list[random_id]
        print(f"\nPerforming search for query: '{query}' with top {k} results...\n")
        search_results = search_faiss(query, model, tokenizer, index, mapping_df, device, k=k)
        results=[]
        results_dict = dict()
        check_unreranked = False
        for res  in search_results:
            results.append(res["answer_chunk"])
            results_dict[res["answer_chunk"]] = res['question']
            print(results_dict[res["answer_chunk"]])
            if results_dict[res["answer_chunk"]] == query:
                check_unreranked = True
        rerank_results = rerank_results_bge(query, results, device="cuda", num=num)

        check_reranked =  False
        question_list = []
        tmp_cnt = -1
        for id,res in enumerate(rerank_results):
            tmp_quest = results_dict[res["answer_chunk"]]
            question_list.append(tmp_quest)
            if tmp_quest == query:
                check_reranked = True
                tmp_cnt = id

        print("RESULTS of 1st search", check_unreranked)
        print("2nd check", check_reranked)
        if check_unreranked:
            if check_reranked:
                indices_reranked_passed_list.append(tmp_cnt)
                indices_reranked_total_list.append(tmp_cnt)
            else: 
                indices_reranked_total_list.append(tmp_cnt)
        print("LEN passed", len(indices_reranked_passed_list))
        print(indices_reranked_passed_list)
        print("LEN total", len(indices_reranked_total_list))
        print(indices_reranked_total_list)
            
    # print(f"\nPerforming search for query: '{query}' with top {k} results...\n")
    # search_results = search_faiss(query, model, tokenizer, index, mapping_df, device, k=k)
    # print("sr 0")
    # print(search_results[0])

    # results=[]
    # results_dict = dict()
    # check_unreranked = False
    # for res  in search_results:
    #     results.append(res["answer_chunk"])
    #     results_dict[res["answer_chunk"]] = res['question']
    #     print(results_dict[res["answer_chunk"]])
    #     if results_dict[res["answer_chunk"]] == query:
    #         check_unreranked = True
    #         print("===================================================")
    #         print("found the question", results_dict[res["answer_chunk"]])
    # print("===================================================")
    # print("RESULTS of 1st search", check_unreranked)
    # rerank_results = rerank_results_bge(query, results, device="cuda", num=10)

    # check_reranked =  False
    # print("this is the list of res")
    # question_list = []
    # tmp_cnt = 0
    # for res in rerank_results:
    #     tmp_cnt += 1
    #     tmp_quest = results_dict[res["answer_chunk"]]
    #     print(tmp_quest)
    #     question_list.append(tmp_quest)
    #     if tmp_quest == query:
    #         print("===================================================")
    #         print("found the question", tmp_cnt,  tmp_quest, res["answer_chunk"])
    #         check_reranked = True
    # print("2nd check", check_reranked)

if __name__ == "__main__":
    main()
