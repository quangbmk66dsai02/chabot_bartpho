import os
import random
import json
from openai import OpenAI
from tqdm import tqdm

def read_vietnamese_text(file_name):
    file_path = f"data/{file_name}.txt"
    with open(file_path, "r", encoding="utf-8-sig") as file:
        text = file.read()
    return text

# Set the API key and model name
MODEL = "gpt-4o-mini"
key = os.getenv('OPENAI_API_KEY')
if not key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
client = OpenAI(api_key=key)

start = 10
end = 2000
sample_size = 50 
unique_random_numbers = random.sample(range(start, end + 1), sample_size)

output_data = []

for id in tqdm(unique_random_numbers):
    print("Currently processing data", id)
    file_content = read_vietnamese_text(str(id))
    noq = 5
    
    # Call the API
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Bạn là một trợ lý chuyên về lịch sử. Hãy tạo ra câu hỏi liên quan đến đoạn mẫu đã được cho. Không hỏi nhưng câu hỏi tại sao, như thế nào và các câu hỏi có câu trả lời dài quá 150 từ"},
            {"role": "user", "content": f"""
            Hãy nghĩ ra {noq} câu hỏi khoảng 5 đến 30 từ liên quan đến đoạn văn bản.
            Data: {file_content}
            Question:
            """},
        ]
    )
    
    response_content = completion.choices[0].message.content.strip()
    
    # Extract questions (assuming they are listed in the response)
    questions = [line.strip() for line in response_content.split("\n") if line.strip()]
    
    # Store results in JSON format
    output_data.append({
        "data": id,
        "questions": questions
    })

# Write output to a JSON file
output_file = "test_data/1000questions.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Questions saved to {output_file}")