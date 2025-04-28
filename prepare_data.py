import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ast import literal_eval

# Read the CSV file
df = pd.read_csv('movies_overview.csv')

# Print column names to verify structure
print("Available columns:", df.columns.tolist())

# Print first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# Convert string representation of list in genre_ids to actual list
df['genre_ids'] = df['genre_ids'].apply(literal_eval)

# Split all data into train and test sets
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Save complete train and test datasets
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print("\nData splitting complete!")
print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")
print("\nColumns in training data:", train_df.columns.tolist())