import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import gradio as gr
import faiss
from openai import OpenAI

# Set OpenAI API key
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)

def get_absolute_path(relative_path):
    """
    Get the absolute path based on the current script's location.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(script_dir, relative_path)

def load_faiss_index_and_mapping(faiss_index_path, mapping_csv_path):
    """
    Load FAISS index and mapping DataFrame.
    """
    index = faiss.read_index(faiss_index_path)
    mapping_df = pd.read_csv(mapping_csv_path, encoding='utf-8-sig')
    return index, mapping_df

def get_query_embedding(model, tokenizer, query, device, max_length=256):
    """
    Generate query embedding using a pretrained model.
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
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask

        return embedding.cpu().numpy().astype('float32')

def search_faiss(query, model, tokenizer, index, mapping_df, device, k=5):
    """
    Search FAISS index with query embedding.
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
    return results

def rerank_results_bge(query, results, device='cuda', num=1):
    """
    Rerank results using BGEM3FlagModel.
    """
    from FlagEmbedding import BGEM3FlagModel

    devices = str(device)
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
        use_fp16=True,
        pooling_method='cls',
        devices=devices
    )
    corpus = results
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

    sparse_similarity = np.array(sparse_similarity)[0]
    dense_similarity = np.array(dense_similarity)[0]
    max_indices = np.argsort(dense_similarity)[::-1][:num]
    results_reranked = [{'dense_score': dense_similarity[i], 'sparse_score': sparse_similarity[i], 'answer_chunk': results[i]} for i in max_indices]
    return results_reranked

def generate_gpt4_mini_response(query, answer_content):
    """
    Generate a response using GPT-4 Mini model based on the query and content.
    """
    MODEL = "gpt-4o-mini-2024-07-18"
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là một trợ lý hỗ trợ QA lịch sử. "
                    "Hãy trả lời câu hỏi dựa trên những thông tin đc cung cấp một cách chi tiết. "
                    "Hãy bỏ qua những thông tin không liên quan. "
                    "Hãy trả về cả vị trí của câu trả lời dựa vào trường Link, nếu không có hãy trả về trường Location. "
                    "Bạn có thể chọn nhiều thông tin để trả lời. Lưu ý chỉ trả lời trên thông tin đã cung cấp, không được dựa vào bất kỳ hiểu biết nào khác."
                )
            },
            {
                "role": "user",
                "content": f"Đây là câu hỏi {query}. Còn đây là list nội dung có thể liên quan đến câu trả lời:\n{answer_content}"
            },
        ]
    )
    response_content = completion.choices[0].message.content
    return response_content

def process_query(query, k=5):
    """
    Main processing function for the query.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = get_absolute_path('../pho_embedding_train/fine-tuned-phobert-embedding-model')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModel.from_pretrained(model_path).to(device)

    faiss_index_path = get_absolute_path('chunked_faiss_index.bin')
    mapping_csv_path = get_absolute_path('chunked_faiss_mapping.csv')
    article_links_path = get_absolute_path('article_links.csv')

    index, mapping_df = load_faiss_index_and_mapping(faiss_index_path, mapping_csv_path)
    search_results = search_faiss(query, model, tokenizer, index, mapping_df, device, k=100)
    results = [res["text_chunk"] for res in search_results]
    reranked_results = rerank_results_bge(query, results, device=device, num=5)

    df = pd.read_csv(article_links_path)
    file_to_link = dict(zip(df['Filename'], df['URL']))

    for res in reranked_results:
        tmp_key = res['answer_chunk']
        for _, data_rec in mapping_df.iterrows():  # Use iterrows to iterate through rows
            if tmp_key == data_rec['text_chunk']:
                res['original_text'] = data_rec['file_name']

    # Generate answer content for GPT-4 Mini
    answer_content = ""
    for res in reranked_results:
        filename = res['original_text']
        original_link = file_to_link.get(filename, "Link not found")
        answer_content += (
            f"Content: {res['answer_chunk']}, "
            f"Dense Score: {res['dense_score']}, "
            f"Sparse Score: {res['sparse_score']}, "
            f"Location: {filename}, "
            f"Link: {original_link}\n\n"
        )
        print(res['answer_chunk'])
    
    # Get GPT-4 Mini response
    response_content = generate_gpt4_mini_response(query, answer_content)
    return response_content

# Gradio interface
interface = gr.Interface(
    fn=process_query,
    inputs=[gr.Textbox(label="Query"), gr.Slider(1, 20, value=5, label="Top-K Results")],
    outputs="text",
    title="FAISS QA Search Vietnam History",
    description="RAG system for QA Vietnam history."
)

if __name__ == "__main__":
    interface.launch(share=True)
