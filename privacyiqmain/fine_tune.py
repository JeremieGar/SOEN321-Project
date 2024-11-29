import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import classification_report


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def load_data(file_path):
    """
    Load and split the dataset into training and validation sets.
    """
    data = pd.read_csv(file_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data["text"].tolist(), data["label"].tolist(), test_size=0.2, random_state=42
    )
    return train_texts, val_texts, train_labels, val_labels

def fine_tune_model(train_texts, train_labels, val_texts, val_labels, model_name="bert-base-uncased"):
    """
    Fine-tune a BERT model for multi-class classification.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)  # 4 classes: 0-3

    # Create PyTorch datasets
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Fine-tuned model saved to './fine_tuned_model'.")

def evaluate_model(model, tokenizer, val_texts, val_labels):
    """
    Evaluate the fine-tuned model on the validation set.
    """
    predictions = []
    for text in val_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities).item()
        predictions.append(predicted_class)

    print("\nClassification Report:")
    print(classification_report(val_labels, predictions, target_names=["No Risk", "Low Risk", "Medium Risk", "High Risk"]))

if __name__ == "__main__":
    # Load the merged dataset
    train_texts, val_texts, train_labels, val_labels = load_data("data/merged_training_data.csv")

    # Fine-tune the model
    fine_tune_model(train_texts, train_labels, val_texts, val_labels)

    # Load the fine-tuned model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
    # Evaluate the model on the validation set
    evaluate_model(model, tokenizer, val_texts, val_labels)

