import torch
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments, AutoTokenizer
from load_preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split
import os

def fine_tune_and_train(dataset_path, model_name="distilbert/distilbert-base-uncased", save_path="trained_model"):
    # Load and preprocess dataset
    df, preprocess = load_and_preprocess(dataset_path, model_name)

    # Split dataset
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

    train_encodings = preprocess(train_data)
    val_encodings = preprocess(val_data)

    # Create Dataset class
    class QADataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    train_dataset = QADataset(train_encodings)
    val_dataset = QADataset(val_encodings)

    # Load model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,  # Reduce epochs slightly
        per_device_train_batch_size=4,  # Smaller batch size to fit in CPU memory
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,  # Log less frequently to avoid overhead
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save checkpoints only at the end of each epoch
        save_total_limit=1,  # Keep only the latest checkpoint to save disk space
        gradient_accumulation_steps=2,  # Simulate larger batch sizes by accumulating gradients
        fp16=False,  # Disable mixed precision as it's only useful on GPUs
        learning_rate=5e-5  # Use a slightly higher learning rate for fewer epochs
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train model
    trainer.train()

    # Save the model locally
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    dataset_path = "qa_model_dataset.json"
    model_name = "distilbert/distilbert-base-uncased"
    save_path = "trained_model"
    fine_tune_and_train(dataset_path, model_name, save_path)
