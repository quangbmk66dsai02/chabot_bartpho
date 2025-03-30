import pandas as pd
import os
import json
from tqdm import tqdm
from openai import OpenAI
# Load the CSV file into a DataFrame
csv_file_path = 'database_building/chunked_text_data.csv'  # Update with your CSV file path
df = pd.read_csv(csv_file_path)
input(f"THIS IS LENGTH DF{len(df)}", )
# Sample 1,000 paragraphs randomly
sample_size = len(df)
sampled_df = df.sample(n=sample_size, random_state=42)  # Setting random_state for reproducibility
key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)

# Define the model to use
MODEL = "gpt-4o-mini"

# Initialize a list to store the output data
output_data = []

# Iterate over the sampled paragraphs and generate questions
for index, row in tqdm(sampled_df.iterrows(), total=sample_size):
    paragraph_id = row['id']
    paragraph_text = row['text_chunk']

    # Define the prompt for the API
    prompt = f"""
    Hãy nghĩ ra 2 câu hỏi khoảng 5 đến 30 từ liên quan đến đoạn văn bản sau.
    Không hỏi những câu hỏi tại sao, như thế nào, hãy sử dụng tên riêng nếu có thể và các câu hỏi có câu trả lời mở.

    Đoạn văn bản: {paragraph_text}

    Câu hỏi:
    """

    # Call the OpenAI API
    try:
        completion = client.chat.completions.create(

            model=MODEL,
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý chuyên về lịch sử."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150
        )
        response_content = completion.choices[0].message.content.strip()
        questions = [q.strip() for q in response_content.split("\n") if q.strip()]

        # Store the results
        output_data.append({
            "paragraph_id": paragraph_id,
            "paragraph_text": paragraph_text,
            "questions": questions
        })
    except Exception as e:
        print(f"Error processing paragraph ID {paragraph_id}: {e}")

# Save the output data to a JSON file
output_file_path = 'test_data/paragraph_full.json'  # Update with your desired output file path
with open(output_file_path, 'a', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Generated questions saved to {output_file_path}")
