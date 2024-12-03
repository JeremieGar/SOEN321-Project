import pandas as pd

def merge_datasets(dataset1, dataset2, output_file):
    """
    Merge two datasets and save to a new CSV file.

    Parameters:
        dataset1 (str): Path to the first dataset (existing training data).
        dataset2 (str): Path to the second dataset (newly labeled data).
        output_file (str): Path to save the merged dataset.
    """
    # Load both datasets
    df1 = pd.read_csv(dataset1)
    df2 = pd.read_csv(dataset2)

    # Concatenate datasets
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Shuffle the merged dataset for better training
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the merged dataset
    merged_df.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file}")

# Example usage
if __name__ == "__main__":
    merge_datasets(
        "data/multi_privacy_policies.csv",
        "data/labeled_policies.csv",
        "data/merged_training_data.csv"
    )
