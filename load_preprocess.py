import json
import pandas as pd
import torch
from transformers import AutoTokenizer

def load_and_preprocess(dataset_path, model_name="distilbert/distilbert-base-uncased"):
# Load dataset with UTF-8 encoding
    with open(dataset_path, "r", encoding="utf-8") as f:  # Specify encoding
        data = json.load(f)

    records = []
    for item in data['data']:
        records.append({
            "id": item["id"],
            "question": item["question"],
            "description": item["description"],
            "context": item["context"],
            "answer_text": item["answers"]["text"][0] if item["answers"]["text"] else "",
            "answer_start": item["answers"]["answer_start"][0] if item["answers"]["answer_start"] else -1
        })

    df = pd.DataFrame(records)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare encodings
    def preprocess(data):
        encodings = tokenizer(
            data['context'].tolist(),
            data['question'].tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        encodings.update({
            "start_positions": torch.tensor(data['answer_start'].tolist()),
            "end_positions": torch.tensor([
                start + len(ans) for start, ans in zip(data['answer_start'], data['answer_text'])
            ])
        })
        return encodings

    return df, preprocess

if __name__ == "__main__":
    dataset_path = "qa_model_dataset.json"
    model_name = "distilbert/distilbert-base-uncased"
    df, preprocess = load_and_preprocess(dataset_path, model_name)
    print(df.head())