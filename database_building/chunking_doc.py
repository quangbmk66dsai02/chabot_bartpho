import os
import pandas as pd
from database_utilities import *  # Ensure this module contains the `chunk_text_by_sentences_with_overlap` function.

# Specify the folder containing text files
folder_path = "data"  # Update with the folder path containing text files.

# Initialize a list to store the chunked data
chunked_data = []

# Define maximum words per chunk
MAX_WORDS_PER_CHUNK = 150  # Adjust based on your embedding model's requirements

# Process each text file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # Only process .txt files
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()  # Read the entire content of the file
        
        # Chunk the text using the chunking utility
        chunks = chunk_text_by_sentences_with_overlap(text, max_words=MAX_WORDS_PER_CHUNK, overlap_words=30)
        
        # Add chunks to the list with metadata
        for chunk in chunks:
            chunked_data.append({
                'file_name': filename,
                'text_chunk': chunk
            })

# Convert the chunked data into a DataFrame
df_chunked = pd.DataFrame(chunked_data)

# Compute word counts for chunks
df_chunked['chunk_word_count'] = df_chunked['text_chunk'].apply(lambda x: len(x.split()))
df_chunked['chunk_sentence_count'] = df_chunked['text_chunk'].apply(lambda x: len(split_into_sentences(x)))

# Save the chunked data to a CSV file
output_csv_path = "database_building/chunked_text_data.csv"
df_chunked.to_csv(output_csv_path, encoding="utf-8-sig", index=False)

# Display statistics and sample chunks
print(f"Total chunks created: {len(df_chunked)}")
print("\nDescriptive Statistics for Chunked Text (Words):")
print(df_chunked['chunk_word_count'].describe())

print("\nDescriptive Statistics for Chunked Text (Sentences):")
print(df_chunked['chunk_sentence_count'].describe())

# Display a sample of the chunked data
random_samples = df_chunked.sample(n=5, random_state=42)  # For reproducibility
print("\nSample Chunked Data:")
print(random_samples)

# Function to display details of a specific chunk
def display_chunk(row, chunk_number):
    print(f"--- Chunk {chunk_number} ---")
    print(f"File Name: {row['file_name']}")
    print(f"Text Chunk: {row['text_chunk']}")
    print(f"Word Count: {row['chunk_word_count']}")
    print(f"Sentence Count: {row['chunk_sentence_count']}")
    print("-" * 80 + "\n")

# Optionally, display the first 10 chunks
first_10_chunks = df_chunked.iloc[:10]
for idx, row in first_10_chunks.iterrows():
    chunk_number = idx + 1
    display_chunk(row, chunk_number)
