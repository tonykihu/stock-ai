import os
import pandas as pd
from ta import add_all_ta_features

def preprocess_data():
    """
    Loads US and Kenya stock data, adds technical indicators,
    and saves processed features to CSV.
    """
    os.makedirs("data/processed", exist_ok=True)

    # Load datasets
    us_data = pd.read_csv("data/us/aapl.csv", parse_dates=["Date"])
    kenya_data = pd.read_csv("data/kenya/nse_latest.csv")

    # Standardize columns
    us_data["Market"] = "US"
    kenya_data["Market"] = "Kenya"
    combined = pd.concat([us_data, kenya_data], ignore_index=True)

    # Save combined raw data
    combined.to_csv("data/processed/combined.csv", index=False)

    # Add technical indicators only for rows that have OHLCV data (US stocks)
    us_only = combined[combined["Market"] == "US"].copy()

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if all(col in us_only.columns for col in required_cols):
        us_only = us_only.dropna(subset=required_cols)
        us_only = add_all_ta_features(
            us_only, open="Open", high="High", low="Low",
            close="Close", volume="Volume"
        )

    # Map ta library column names to our standard names
    column_mapping = {
        "momentum_rsi": "rsi_14",
        "trend_macd": "macd",
        "trend_sma_fast": "sma_50",
        "volume_obv": "volume_obv"
    }

    for ta_name, our_name in column_mapping.items():
        if ta_name in us_only.columns:
            us_only = us_only.rename(columns={ta_name: our_name})

    # Keep key features
    features = ["rsi_14", "macd", "sma_50", "volume_obv"]
    available_features = [f for f in features if f in us_only.columns]
    keep_cols = available_features + ["Close", "Market", "Date"]
    us_only[keep_cols].to_csv("data/processed/features.csv", index=False)
    print(f"Features saved with columns: {keep_cols}")

if __name__ == "__main__":
    preprocess_data()
