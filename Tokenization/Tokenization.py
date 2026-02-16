text = "I love programming"
tokens = text.split()
print(tokens)

import h5py
import pandas as pd
from transformers import BertTokenizer

# 1. Define the file path for your .nc4 dataset
# Updated to use the name of the uploaded file
file_path = "yield_1993.nc4"

# 2. Open the .nc4 file using h5py
with h5py.File(file_path, 'r') as file:
    # 3. Print available keys (datasets) in the NetCDF file
    print("Available Keys (Datasets):")
    for key in file.keys():
        print(key)

    # 4. Example: Access the first dataset (adjust if necessary)
    dataset_name = list(file.keys())[0]  # Get the first dataset key
    dataset = file[dataset_name]

    # 5. Print dataset shape and the first few values (to inspect)
    print(f"\nDataset '{dataset_name}' shape: {dataset.shape}")
    print(f"First few values in '{dataset_name}':")
    print(dataset[:10])  # Show first 10 values

    # 6. Convert the dataset into a Pandas DataFrame (if applicable, for 2D datasets)
    data_array = dataset[:]

    if len(data_array.shape) == 2:  # Check if the dataset is 2D (rows and columns)
        df = pd.DataFrame(data_array)
        print("\nData loaded into DataFrame:")
        print(df.head())
    else:
        print("\nDataset is not 2D. Cannot convert to DataFrame.")
        # Initialize an empty DataFrame if not 2D to prevent errors in subsequent steps
        df = pd.DataFrame()

# 7. Tokenize the text data (if applicable)
# Assume we have a 'text' column for tokenization (adjust column name as necessary)
# For example, let's assume the dataset has text-like data stored in 'column_name'
# Here we will try to tokenize data if it is available

# Initialize the BERT tokenizer (only if df is not empty and we might proceed with tokenization)
if not df.empty:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_text(text):
        """Function to tokenize text"""
        # Ensure text is a string to avoid errors with non-string types
        if isinstance(text, str):
            return tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128
            )
        return None # Return None or handle non-string types as appropriate

    # Check if the dataframe contains text-like columns and tokenize them
    # This part might need adjustment based on the actual content of 'yield_1993.nc4'
    # For demonstration, let's assume if df is not empty, we can try to tokenize some column
    # If the dataset has no text columns, this will be skipped or need further modification
    # Placeholder for a potential text column, you might need to adjust this after inspecting dataset keys
    potential_text_column = None
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].apply(lambda x: isinstance(x, str)).any():
            potential_text_column = col
            break

    if potential_text_column:
        df['tokenized_output'] = df[potential_text_column].apply(tokenize_text)
        print("\nTokenized Text (First 5 rows):")
        print(df['tokenized_output'].head())
    else:
        print("\nNo suitable text column found for tokenization.")

# 8. Save processed data (tokenized) to a new CSV file
# Updated output path for Colab environment
if not df.empty:
    output_csv_path = "processed_yield_data.csv"
    df.to_csv(output_csv_path, index=False)

    print(f"\nProcessed data saved to: {output_csv_path}")
else:
    print("\nNo data to save to CSV as DataFrame was not created or was empty.")
