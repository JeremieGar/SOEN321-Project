import torch
from privacyiqmain.inference import load_model, run_inference


def test_model():
    model, tokenizer = load_model()
    examples = [
        "We may collect your data for advertising purposes.",
        "Your data will not be shared with third parties without your consent.",
        "We respect your privacy and will not sell your data to any third parties.",
        "We reserve the right to share your personal information with our partners.",
        "Your data will only be used for improving your user experience on our platform.",
        "We will sell your data to partners for marketing purposes.",
        "Your browsing history may be tracked for product recommendations."
    ]

    for text in examples:
        print(f"\nTesting on example: '{text}'")
        probabilities = run_inference(text, model, tokenizer)
        print("Probabilities:", probabilities)
        # Interpreting the output
        predicted_class = torch.argmax(probabilities).item()
        labels = ["Low Risk", "High Risk"]
        print(f"Predicted Class: {labels[predicted_class]}")


if __name__ == "__main__":
    test_model()
