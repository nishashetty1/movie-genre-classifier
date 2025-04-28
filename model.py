import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

class MovieGenreEnsembleClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=random_state
        )
        self.gb_classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
        self.label_encoder = LabelEncoder()
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        
    def preprocess_data(self, X):
        """
        Preprocess the text data using TF-IDF vectorization
        """
        return self.tfidf.transform(X)
    
    def fit(self, X, y):
        """
        Train both Random Forest and Gradient Boosting models
        """
        # Encode the target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Preprocess the features
        X_processed = self.preprocess_data(X)
        
        # Train Random Forest
        self.rf_classifier.fit(X_processed, y_encoded)
        
        # Train Gradient Boosting
        self.gb_classifier.fit(X_processed, y_encoded)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using ensemble of both models
        """
        X_processed = self.tfidf.transform(X)
        
        # Get predictions from both models
        rf_pred = self.rf_classifier.predict_proba(X_processed)
        gb_pred = self.gb_classifier.predict_proba(X_processed)
        
        # Combine predictions (simple averaging)
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        # Get the final predictions
        final_predictions = np.argmax(ensemble_pred, axis=1)
        
        # Convert numeric predictions back to original labels
        return self.label_encoder.inverse_transform(final_predictions)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the ensemble model
        """
        predictions = self.predict(X_test)
        return classification_report(y_test, predictions)

def main():
    # Example usage
    try:
        # Load your movie data here
        # Assuming format: df with 'text' (movie description) and 'genre' columns
        df = pd.read_csv('your_movie_data.csv')
        
        # Split features and target
        X = df['text'].values
        y = df['genre'].values
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train the ensemble model
        model = MovieGenreEnsembleClassifier()
        model.fit(X_train, y_train)
        
        # Evaluate the model
        print("Model Evaluation:")
        print(model.evaluate(X_test, y_test))
        
        # Example prediction
        sample_movies = [
            "A thrilling action movie with explosive scenes and car chases",
            "A romantic story about two people falling in love in Paris"
        ]
        predictions = model.predict(sample_movies)
        print("\nSample Predictions:")
        for movie, genre in zip(sample_movies, predictions):
            print(f"Movie: {movie}\nPredicted Genre: {genre}\n")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()