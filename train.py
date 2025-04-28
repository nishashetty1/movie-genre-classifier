import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from model import MovieGenreEnsembleClassifier

def load_data(train_path, test_path):
    """
    Load and prepare the movie dataset from train and test files
    """
    try:
        # Load train and test datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Testing data shape: {test_df.shape}")
        
        return train_df, test_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def prepare_data(train_df, test_df):
    """
    Prepare training and testing datasets
    """
    # Assuming your CSVs have 'text' and 'genre' columns
    # If they have different column names, modify these accordingly
    X_train = train_df['text'].values
    y_train = train_df['genre'].values
    X_test = test_df['text'].values
    y_test = test_df['genre'].values
    
    return X_train, X_test, y_train, y_test

def save_model(model, model_path):
    """
    Save the trained model to disk
    """
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model successfully saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def train_model(train_path, test_path, model_save_path='models', model_filename='movie_genre_classifier.pkl'):
    """
    Main training function
    """
    # Create models directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path)
    if train_df is None or test_df is None:
        return
    
    # Print initial data information
    print("\nTraining Dataset Information:")
    print(f"Total training samples: {len(train_df)}")
    print("\nGenre distribution in training data:")
    print(train_df['genre'].value_counts())
    
    # Prepare data
    print("\nPreparing training and testing data...")
    X_train, X_test, y_train, y_test = prepare_data(train_df, test_df)
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Initialize and train the model
    print("\nInitializing and training the ensemble model...")
    model = MovieGenreEnsembleClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    
    # Save the model
    model_path = os.path.join(model_save_path, model_filename)
    print("\nSaving the model...")
    save_model(model, model_path)
    
    # Example predictions
    print("\nTesting with sample movies...")
    sample_movies = [
        "A thrilling action movie with explosive scenes and car chases",
        "A romantic story about two people falling in love in Paris",
        "A documentary about nature and wildlife in Africa",
        "A comedy about a family vacation gone wrong"
    ]
    
    predictions = model.predict(sample_movies)
    print("\nSample Predictions:")
    for movie, genre in zip(sample_movies, predictions):
        print(f"\nMovie: {movie}")
        print(f"Predicted Genre: {genre}")

if __name__ == "__main__":
    # Paths to your data files
    train_path = "data/train_data.csv"
    test_path = "data/test_data.csv"
    
    print("=== Movie Genre Classification Model Training ===")
    print(f"Starting training process at: {pd.Timestamp.now()}")
    
    train_model(train_path, test_path)
    
    print(f"\nTraining completed at: {pd.Timestamp.now()}")