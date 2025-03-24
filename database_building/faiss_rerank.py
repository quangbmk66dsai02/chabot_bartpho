import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import pandas as pd
import numpy as np
from openai import OpenAI

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def get_absolute_path(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

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
                'text_chunk': mapping_df.iloc[idx]['text_chunk'],
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
    parser = argparse.ArgumentParser(description="FAISS Search with Fine-Tuned PhoBERT")
    parser.add_argument('--query', type=str, default="Lý Công Uẩn sinh ra ở đâu?", help='Search query string')
    parser.add_argument('--k', type=int, default=50, help='Number of nearest neighbors to retrieve')
    parser.add_argument('--faiss_index', type=str, default='database_building/chunked_faiss_index.bin', help='Path to FAISS index file')
    parser.add_argument('--mapping_csv', type=str, default='database_building/chunked_faiss_mapping.csv', help='Path to mapping CSV file')
    parser.add_argument('--model_path', type=str, default=get_absolute_path('../pho_embedding_train/fine-tuned-phobert-embedding-model'))
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the fine-tuned PhoBERT model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading the fine-tuned PhoBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    print("Model and tokenizer loaded successfully.")

    # Load the FAISS index and mapping
    print("Loading FAISS index and mapping DataFrame...")
    index, mapping_df = load_faiss_index_and_mapping(args.faiss_index, args.mapping_csv)
    # Perform the search
    query = args.query
    k = args.k
    print(f"\nPerforming search for query: '{query}' with top {k} results...\n")
    search_results = search_faiss(query, model, tokenizer, index, mapping_df, device, k=k)

    results=  []
    for res  in search_results:
        results.append(res["text_chunk"])
    rerank_results = rerank_results_bge(query, results, device="cuda", num=5)
    for res in rerank_results:
        tmp_key = res['answer_chunk']
        for _, data_rec in mapping_df.iterrows():  # Use iterrows to iterate through rows
            if tmp_key == data_rec['text_chunk']:
                res['original_text'] = data_rec['file_name']
                
    file_path = "database_building/article_links.csv"
    df = pd.read_csv(file_path)

    file_to_link = dict(zip(df['Filename'], df['URL']))

    # Generate the answer content
    answer_content = ""
    for id, res in enumerate(rerank_results):
        # Get the filename from res['original_text']
        filename = res['original_text']
        
        # Get the corresponding link using the mapping
        original_link = file_to_link.get(filename, "Link not found")
        
        # Format the answer content
        answer_content += f"{id}. Content: {res['answer_chunk']}. Location: {filename}, Link: {original_link} + \n"
        print(answer_content)


    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # # Load LLaMA 3.2 model and tokenizer
    # llama_model_path = "meta-llama/Llama-3.2-3B-Instruct"  # Change to local path if using a downloaded model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Loading LLaMA 3.2 model...")
    # llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    # llama_model = AutoModelForCausalLM.from_pretrained(llama_model_path, torch_dtype=torch.float16, device_map="auto")
    # print("LLaMA model loaded.")

    MODEL = "gpt-4o-mini"
    key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=key)

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"""You are an assistant that answers questions based solely on the provided documents. 
                Do not use any external knowledge or information not contained within the provided data. 
                If the answer is not found in the documents, respond with 'Không có thông tin trong cơ sở dữ liệu.'
                Also include link and location of text evidence in the answer.
                Example:
                câu hỏi: Đại tướng Võ Nguyên Giáp sinh ra ở đâu?
                nội dung cung cấp: 0. Content Võ Nguyên Giáp sinh ra ở Quảng Bình. Location: abc.txt, Link: xyz
                câu trả lời: Đại tướng Võ NGuyên Giáp sinh ra ở Quảng Bình. Location: abc.txt, Link: xyz
                câu hỏi: Vị vua nào được dân gian mô tả với "gương mặt sắt đen sì"?
                nội dung cung cấp: 
                0. Mai Hắc Đế là vị vua anh minh. Location abc.txt, Link xyz, 
                1. Nhà Trần được thành lập năm ... Location abc.txt, Link xyz.
                câu trả lời: Không tìm được thông tin liên quan """},
            {"role": "user", "content": f"""câu hỏi: {query}. 
                nội dung cung cấp: {answer_content}
                câu trả lời:"""},
        ]
    )
    print("==============THIS IS THE RETRIEVED CONTENT===========================", answer_content)
    response_content = completion.choices[0].message.content
    print("============THIS IS THE RESPONSE CONTENT", response_content)

if __name__ == "__main__":
    main()
