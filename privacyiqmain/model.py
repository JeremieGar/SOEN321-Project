from transformers import AutoModelForSequenceClassification

def load_model(num_labels=5, model_name="bert-base-uncased"):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
