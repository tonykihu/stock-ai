import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    """Retrain both hybrid and technical models"""
    
    # Make sure the models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Retrain hybrid model
    retrain_hybrid_model()
    
    # Retrain technical model  
    retrain_technical_model()

def retrain_hybrid_model():
    """Retrain the hybrid model (technical + sentiment)"""
    model_path = "models/hybrid_model.pkl"
    
    print("Retraining hybrid model...")
    
    try:
        # Load technical features and sentiment
        tech_data = pd.read_csv("data/processed/features.csv")
        sentiment = pd.read_csv("data/processed/news_sentiment.csv")
        
        # Merge data (assuming same dates)
        merged = pd.merge(tech_data, sentiment, on="Date")
        
        # Create target if it doesn't exist
        if "Target" not in merged.columns:
            merged["Target"] = (merged["Close"].shift(-1) > merged["Close"]).astype(int)
        
        # Prepare features
        X = merged[["rsi_14", "sma_50", "Score"]].dropna()
        y = merged["Target"].dropna()
        
        # Align X and y (in case of different lengths due to dropna)
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
        
        # Train hybrid model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save model
        joblib.dump(model, model_path)
        print(f"Hybrid model retrained and saved to {model_path}")
        
        # Print some basic info
        print(f"Training data shape: {X.shape}")
        if len(X) > 0:
            print(f"Model accuracy on training data: {model.score(X, y):.4f}")
        
    except FileNotFoundError as e:
        print(f"Required data files not found for hybrid model: {e}")
        print("Skipping hybrid model retraining...")
    except Exception as e:
        print(f"Error retraining hybrid model: {e}")

def retrain_technical_model():
    """Retrain the technical model (technical features only)"""
    model_path = "models/technical_model.pkl"
    
    print("Retraining technical model...")
    
    try:
        # Load features
        data = pd.read_csv("data/processed/features.csv").dropna()
        
        # Create target: 1 if price rises next day, else 0
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        
        # Remove the last row (no target due to shift)
        data = data.dropna()
        
        # Split data
        X = data[["rsi_14", "sma_50", "volume_obv"]]
        y = data["Target"]
        
        if len(X) > 10:  # Ensure we have enough data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            
            print(f"Technical model - Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
            
            # Save model
            joblib.dump(model, model_path)
            print(f"Technical model retrained and saved to {model_path}")
            
        else:
            print("Not enough data to retrain technical model")
            
    except FileNotFoundError as e:
        print(f"Required data files not found for technical model: {e}")
        print("Skipping technical model retraining...")
    except Exception as e:
        print(f"Error retraining technical model: {e}")

if __name__ == "__main__":
    main()
