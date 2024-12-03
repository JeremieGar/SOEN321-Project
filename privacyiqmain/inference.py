import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model():
    """
    Load the fine-tuned model and tokenizer.
    """
    model_name = "./fine_tuned_model"  # Path to the fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def run_inference(text, model, tokenizer):
    """
    Perform inference on the input text using the fine-tuned model.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probabilities


if __name__ == "__main__":
    # Example input text
    sample_text = [
        "We may share your personal information with third parties.",
        "Your data will not be shared without your explicit consent.",
        "We reserve the right to sell your data to advertisers."
    ]

    # Load model and tokenizer
    model, tokenizer = load_model()

    # Run inference on sample texts
    for text in sample_text:
        probabilities = run_inference(text, model, tokenizer)
        predicted_class = torch.argmax(probabilities).item()
        print(f"Text: {text}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Probabilities: {probabilities}\n")
