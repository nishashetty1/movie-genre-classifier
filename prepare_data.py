import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ast import literal_eval

# Read the CSV file
df = pd.read_csv('movies_overview.csv')

# Print column names to verify structure
print("Available columns:", df.columns.tolist())

# Since the last column contains the genres, we'll use position-based indexing
# Let's look at the first few rows to verify the data
print("\nFirst few rows of the dataset:")
print(df.head())

# Convert string representation of list to actual list in the last column
df.iloc[:, -1] = df.iloc[:, -1].apply(literal_eval)

# Create features (X) from title (first column) and overview (second column)
df['text'] = df.iloc[:, 0] + ' ' + df.iloc[:, 1]
X = df['text']

# Create target variable (y) from genres (last column)
y = df.iloc[:, -1]

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=None  # Since we have multilabel data, we can't use stratify
)

# Save the splits for later use
train_data = pd.DataFrame({
    'text': X_train,
    'genres': y_train
})

test_data = pd.DataFrame({
    'text': X_test,
    'genres': y_test
})

# Save to CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("\nData splitting complete!")
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")