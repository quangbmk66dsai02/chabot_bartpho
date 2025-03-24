import pandas as pd
import re

def clean_unfinished_sentences(text):
    """Removes unfinished sentences at the start and end of a chunk."""
    # Ensure the text starts with a complete sentence
    sentences = re.split(r'([.!?])\s+', text)  # Split while keeping punctuation
    cleaned_sentences = []
    
    # Remove unfinished start (skip if it doesn't start with a capital letter)
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        if sentence and sentence[0].isupper():  # Keep if it starts with a capital letter
            cleaned_sentences.append(sentence + sentences[i + 1])  # Add punctuation back

    # Join back the cleaned sentences
    cleaned_text = " ".join(cleaned_sentences).strip()
    return cleaned_text

# Load the CSV file
csv_file = "database_building/chunked_text_data.csv"  # Change this to your actual CSV file
df = pd.read_csv(csv_file)

# Assuming the text chunks are stored in a column named 'text_chunk'
df['cleaned_chunk'] = df['text_chunk'].apply(clean_unfinished_sentences)

# Save the cleaned data to a new CSV file
df.to_csv("cleaned_text_chunks.csv", index=False)

print("Finished cleaning and saved to cleaned_chunks.csv")
