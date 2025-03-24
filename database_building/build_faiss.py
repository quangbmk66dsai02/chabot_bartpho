import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import faiss
from tqdm import tqdm
import numpy as np

# -------------------------------------------
# Configuration
# -------------------------------------------

# Paths
model_name = 'vinai/bartpho-word'  # Use 'vinai/bartpho-syllable' if you prefer the syllable-level model
chunked_csv_path = 'database_building/chunked_text_data.csv'  # Path to your new CSV file
faiss_index_path = 'database_building/chunked_faiss_index.bin'
mapping_csv_path = 'database_building/chunked_faiss_mapping.csv'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------------------------
# Load the Fine-Tuned PhoBERT Model and Tokenizer
# -------------------------------------------

print("Loading the bartpho model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModel.from_pretrained(model_name)
model.to(device)
model.eval()
print("Model and tokenizer loaded successfully.")

# -------------------------------------------
# Load Your Chunked Data
# -------------------------------------------

print(f"Loading chunked data from '{chunked_csv_path}'...")
df_chunked = pd.read_csv(chunked_csv_path, encoding='utf-8-sig')
print(f"Loaded {len(df_chunked)} chunks.")
print("Sample chunk:")
print(df_chunked.iloc[0])

# -------------------------------------------
# Generate Embeddings for Each Chunk
# -------------------------------------------

class ChunkDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',  # Padding handled in collate_fn
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Shape: (max_length)
            'attention_mask': encoding['attention_mask'].squeeze()  # Shape: (max_length)
        }

def generate_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
            
            # Mean Pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask  # Shape: (batch_size, hidden_size)
            
            # Convert to CPU and NumPy
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
    
    # Concatenate all batches
    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings

# Create the dataset and dataloader
print("Preparing dataset and dataloader for embedding generation...")
texts = df_chunked['text_chunk'].tolist()  # Adjust to match your column name
dataset = ChunkDataset(texts, tokenizer, max_length=256)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # Adjust batch_size based on your GPU memory

# Generate embeddings
print("Generating embeddings for each chunk...")
embeddings = generate_embeddings(model, dataloader, device)
print(f"Generated embeddings shape: {embeddings.shape}")

# -------------------------------------------
# Create and Configure a FAISS Index
# -------------------------------------------

print("Creating and configuring the FAISS index...")
d = embeddings.shape[1]  # e.g., 768 for PhoBERT

# Create FAISS index for exact search
index = faiss.IndexFlatL2(d)

# Add embeddings to the index
index.add(embeddings)
print(f"FAISS index contains {index.ntotal} vectors.")

# -------------------------------------------
# Save the FAISS Index and Mapping
# -------------------------------------------

print("Saving the FAISS index and mapping DataFrame...")

# Save the FAISS index
faiss.write_index(index, faiss_index_path)
print(f"FAISS index saved to '{faiss_index_path}'")

# Prepare mapping DataFrame
df_chunked.reset_index(drop=True, inplace=True)

# Save the mapping DataFrame
df_chunked.to_csv(mapping_csv_path, index=False, encoding='utf-8-sig')
print(f"Mapping DataFrame saved to '{mapping_csv_path}'")
