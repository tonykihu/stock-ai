import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    """Retrain both hybrid and technical models."""
    os.makedirs("models", exist_ok=True)

    retrain_hybrid_model()
    retrain_technical_model()

def retrain_hybrid_model():
    """Retrain the hybrid model (technical + sentiment)."""
    model_path = "models/hybrid_model.pkl"
    print("Retraining hybrid model...")

    try:
        tech_data = pd.read_csv("data/processed/features.csv")
        sentiment = pd.read_csv("data/processed/news_sentiment.csv")

        # Average sentiment per date (in case multiple headlines per day)
        sentiment_daily = sentiment.groupby("Date", as_index=False)["sentiment_score"].mean()

        # Left merge so all feature rows are kept; fill missing sentiment with neutral 0.5
        merged = pd.merge(tech_data, sentiment_daily, on="Date", how="left")
        merged["sentiment_score"] = merged["sentiment_score"].fillna(0.5)

        # Create target if it doesn't exist
        if "Target" not in merged.columns:
            merged["Target"] = (merged["Close"].shift(-1) > merged["Close"]).astype(int)

        # Drop rows with NaN (from shift and missing technical indicators during warmup)
        merged = merged.dropna(subset=["rsi_14", "sma_50", "sentiment_score", "Target"])

        # Prepare features (aligned X and y from the same DataFrame)
        X = merged[["rsi_14", "sma_50", "sentiment_score"]]
        y = merged["Target"]

        if len(X) < 10:
            print("Not enough data to retrain hybrid model")
            return

        # Train hybrid model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Save model
        joblib.dump(model, model_path)
        print(f"Hybrid model retrained and saved to {model_path}")
        print(f"Training data shape: {X.shape}")
        print(f"Model accuracy on training data: {model.score(X, y):.4f}")

    except FileNotFoundError as e:
        print(f"Required data files not found for hybrid model: {e}")
        print("Skipping hybrid model retraining...")
    except Exception as e:
        print(f"Error retraining hybrid model: {e}")

def retrain_technical_model():
    """Retrain the technical model (technical features only)."""
    model_path = "models/technical_model.pkl"
    print("Retraining technical model...")

    try:
        data = pd.read_csv("data/processed/features.csv").dropna()

        # Create target: 1 if price rises next day, else 0
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

        # Remove the last row (NaN target from shift)
        data = data.iloc[:-1]

        # Split data
        X = data[["rsi_14", "sma_50", "volume_obv"]]
        y = data["Target"]

        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            print(f"Technical model - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")

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
