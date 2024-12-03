import pandas as pd
from sklearn.metrics import classification_report
from privacyiqmain.inference import load_model, run_inference
import torch

def main():
    print("Welcome to PrivacyIQ")
    model, tokenizer = load_model()

    test_data = pd.read_csv("data/test_data.csv")
    texts = test_data["text"].tolist()
    true_labels = test_data["label"].tolist()

    # Define class labels
    labels = ["No Risk", "Low Risk", "Medium Risk", "High Risk", "Critical Risk"]

    predicted_labels = []
    for text in texts:
        probabilities = run_inference(text, model, tokenizer)
        predicted_class = torch.argmax(probabilities).item()
        predicted_labels.append(predicted_class)

    # Dynamically adjust labels and target names based on the test set
    present_classes = sorted(set(true_labels))
    present_labels = [labels[i] for i in present_classes]

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, labels=present_classes, target_names=present_labels))

if __name__ == "__main__":
    main()
