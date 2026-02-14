import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.features import TECHNICAL_FEATURES, HYBRID_FEATURES
from utils.evaluation import walk_forward_validate, save_metrics


def main():
    """Retrain both hybrid and technical models."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    retrain_technical_model()
    retrain_hybrid_model()


def retrain_technical_model():
    """Retrain the technical model with walk-forward validation."""
    model_path = "models/technical_model.pkl"
    print("Retraining technical model...")

    try:
        data = pd.read_csv("data/processed/features.csv")
        data = data[data["Ticker"] == "AAPL"].copy()
        data = data.sort_values("Date").reset_index(drop=True)

        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        data = data.dropna(subset=TECHNICAL_FEATURES + ["Target"])

        X = data[TECHNICAL_FEATURES]
        y = data["Target"]

        if len(X) < 100:
            print(f"Not enough data ({len(X)} rows). Skipping.")
            return

        model_params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

        # Evaluate with walk-forward CV
        cv_results = walk_forward_validate(
            RandomForestClassifier, model_params, X, y, n_splits=5
        )
        agg = cv_results["aggregate"]
        print(f"  CV Mean Accuracy: {agg['mean_accuracy']:.4f}, F1: {agg['mean_f1']:.4f}")

        # Train final model on all data
        model = RandomForestClassifier(**model_params)
        model.fit(X, y)

        joblib.dump(model, model_path)
        save_metrics(cv_results, "technical")
        print(f"  Technical model saved ({len(X)} samples)")

    except FileNotFoundError as e:
        print(f"  Data not found: {e}")
    except Exception as e:
        print(f"  Error: {e}")


def retrain_hybrid_model():
    """Retrain the hybrid model with walk-forward validation."""
    model_path = "models/hybrid_model.pkl"
    print("Retraining hybrid model...")

    try:
        tech_data = pd.read_csv("data/processed/features.csv")

        try:
            sentiment = pd.read_csv("data/processed/news_sentiment.csv")
            group_cols = ["Date", "Ticker"] if "Ticker" in sentiment.columns else ["Date"]
            sentiment_daily = sentiment.groupby(group_cols, as_index=False)["sentiment_score"].mean()
            merge_cols = [c for c in group_cols if c in tech_data.columns]
            merged = pd.merge(tech_data, sentiment_daily, on=merge_cols, how="left")
        except FileNotFoundError:
            print("  No sentiment data — using neutral 0.5")
            merged = tech_data.copy()

        merged["sentiment_score"] = merged["sentiment_score"].fillna(0.5) if "sentiment_score" in merged.columns else 0.5

        merged = merged[merged["Ticker"] == "AAPL"].copy()
        merged = merged.sort_values("Date").reset_index(drop=True)
        merged["Target"] = (merged["Close"].shift(-1) > merged["Close"]).astype(int)
        merged = merged.dropna(subset=HYBRID_FEATURES + ["Target"])

        X = merged[HYBRID_FEATURES]
        y = merged["Target"]

        if len(X) < 100:
            print(f"  Not enough data ({len(X)} rows). Skipping.")
            return

        model_params = {"n_estimators": 100, "random_state": 42}

        cv_results = walk_forward_validate(
            RandomForestClassifier, model_params, X, y, n_splits=5
        )
        agg = cv_results["aggregate"]
        print(f"  CV Mean Accuracy: {agg['mean_accuracy']:.4f}, F1: {agg['mean_f1']:.4f}")

        model = RandomForestClassifier(**model_params).fit(X, y)

        joblib.dump(model, model_path)
        save_metrics(cv_results, "hybrid")
        print(f"  Hybrid model saved ({len(X)} samples)")

    except FileNotFoundError as e:
        print(f"  Data not found: {e}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
