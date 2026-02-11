# scripts/preprocess2.py
import os
import pandas as pd
import numpy as np

def clean_historical_data(file_path):
    """
    Cleans raw historical data (handle missing values, outliers)
    and adds technical indicators for Kenya NSE data.
    """
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(file_path)

    # Handle missing data
    df = df.ffill()

    # Remove outliers (prices 3 std deviations away from mean)
    df = df[(df["Price"] - df["Price"].mean()).abs() <= 3 * df["Price"].std()]

    # Add technical indicators
    df["sma_50"] = df["Price"].rolling(50).mean()

    # RSI calculation with safe division
    price_diff = df["Price"].diff(1)
    gain = price_diff.clip(lower=0).rolling(14).mean()
    loss = price_diff.clip(upper=0).abs().rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df.to_csv("data/processed/kenya_processed.csv", index=False)
    print(f"Processed {len(df)} rows and saved to data/processed/kenya_processed.csv")
    return df

if __name__ == "__main__":
    clean_historical_data("data/kenya/nse_historical.csv")
