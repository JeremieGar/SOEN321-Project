import pandas as pd
from privacyiqmain.label_policies import assign_label

def preprocess_policy_file(policy_text):
    """
    Preprocess the privacy policy text, splitting it into sentences, analyzing, and returning labeled results.
    This version labels only the relevant sentences containing keywords.
    """
    # Split the input text into sentences
    sentences = policy_text.split(".")  # Split by period to get individual sentences
    processed_data = []

    # Iterate through each policy sentence
    for sentence in sentences:
        # Check if the sentence contains any keyword, and label only if it does
        if any(keyword in sentence.lower() for keyword in ["personal data", "third parties", "cookies", "tracking", "selling data", "opt-out", "consent", "sharing data"]):
            # Assign the label only if a keyword is found
            label = assign_label(sentence)
            if label is not None:  # Only append labeled sentences
                processed_data.append({"text": sentence.strip(), "label": label})

    # Return the processed data (a list of sentences with their labels)
    return processed_data
