import pandas as pd
import csv  # To handle quoting issues in CSVs


def clean_text(text):
    """
    Clean up text by removing unwanted characters, such as newline, tab, non-breaking spaces, and quotes.
    """
    if not isinstance(text, str):  # If it's not a string, convert it to a string
        text = str(text)  # Convert any non-string values to string

    # Remove special characters like \n, \t, and non-breaking spaces (\xa0)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\xa0', ' ')

    # Strip extra spaces and quotes from the beginning and end of the string
    return text.strip().strip('"')  # Strip quotes specifically here

def assign_label(text):
    """
    Assign a label to a sentence based on keyword matching and context analysis.
    Labels:
    0 - No Risk
    1 - Low Risk
    2 - Medium Risk
    3 - High Risk
    """
    text = clean_text(text).lower()  # Clean and normalize text

    # **No Risk**: Data sharing is explicitly controlled or not happening
    no_risk_keywords = [
        "will not be shared", "will not sell", "explicit consent", "protected by law",
        "strictly confidential", "data control", "restricted access", "no third-party sharing",
        "user approval", "opt-in consent", "consented", "user rights", "privacy guarantee",
        "secure storage", "compliance"
    ]

    # **Low Risk**: Data shared under controlled conditions
    low_risk_keywords = [
        "may share", "anonymized information", "third-party partners", "shared with affiliates",
        "restricted sharing", "partner network", "business purposes", "data processing",
        "operational use", "statistical analysis", "research purposes", "aggregated data",
        "non-personal data", "controlled sharing", "contractual obligations"
    ]

    # **Medium Risk**: User data is tracked or used for personalized recommendations
    medium_risk_keywords = [
        "browsing activity", "personalized recommendations", "cookies", "user tracking",
        "behavioral data", "targeted advertising", "profiling", "usage patterns", "session data",
        "user preferences", "marketing analysis", "data profiling", "tracking technology",
        "data for improvement", "platform analytics"
    ]

    # **High Risk**: Data sold or shared for marketing without adequate consent
    high_risk_keywords = [
        "sell your data", "third-party vendors", "marketing purposes", "shared without consent",
        "data resale", "commercial use", "unauthorized sharing", "data broker", "non-consensual sharing",
        "profiling for marketing", "external advertising", "partner marketing", "data leakage",
        "third-party sales", "unauthorized access"
    ]

    # Track the risk level based on the presence of keywords
    risk_level = 0  # Start with No Risk by default

    # Check for high-risk terms first (priority)
    if any(keyword in text for keyword in high_risk_keywords):
        return 3  # High Risk: if any high-risk keywords are found, return High Risk immediately

    # Check for medium-risk terms if no high-risk terms found
    elif any(keyword in text for keyword in medium_risk_keywords):
        risk_level = max(risk_level, 2)  # Medium Risk

    # Check for low-risk terms if no high or medium-risk terms found
    elif any(keyword in text for keyword in low_risk_keywords):
        risk_level = max(risk_level, 1)  # Low Risk

    # Finally, check for no-risk terms if none of the higher levels were found
    elif any(keyword in text for keyword in no_risk_keywords):
        risk_level = max(risk_level, 0)  # No Risk

    return risk_level

def label_policies(input_file, output_file):
    """
    Label the sentences in the processed data and save the labeled dataset.
    """
    # Load the processed policies file
    df = pd.read_csv(input_file)

    # Display the first few rows of the data for inspection
    print("Original DataFrame:")
    print(df.head())

    # Apply the label assignment
    df["label"] = df["text"].apply(assign_label)

    # Save the labeled dataset to a new CSV file with proper handling of commas and quotes
    # Use escapechar to properly handle commas and quotes inside text
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    print(f"Labeled policies saved to {output_file}")

    # Manually remove quotes from the 'text' column in the saved CSV file
    remove_quotes(output_file)


def remove_quotes(file_path):
    """
    This function removes the quotes from the saved CSV file.
    """
    # Read the CSV file again after saving
    df = pd.read_csv(file_path)

    # Manually strip quotes from the text column
    df["text"] = df["text"].apply(lambda x: str(x).replace('"', '').strip())

    # Save the cleaned CSV back, ensuring no additional quoting is done
    df.to_csv(file_path, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')  # Ensure no quotes in the final CSV
    print(f"Quotes removed and file saved to {file_path}")


# Example usage
if __name__ == "__main__":
    label_policies("data/processed_policies.csv", "data/labeled_policies.csv")
