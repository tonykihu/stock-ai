# scripts/preprocess.py
import pandas as pd

def clean_historical_data(file_path):
    """
    Cleans raw historical data (handle missing values, outliers).
    """
    df = pd.read_csv(file_path)
    
    # Handle missing data
    df.fillna(method="ffill", inplace=True)  # Forward-fill gaps
    
    # Remove outliers (e.g., prices 3 std deviations away from mean)
    df = df[(df["Price"] - df["Price"].mean()).abs() <= 3 * df["Price"].std()]
    
    # Add technical indicators
    df["sma_50"] = df["Price"].rolling(50).mean()
    df["rsi_14"] = 100 - (100 / (1 + (df["Price"].diff(1).clip(lower=0).rolling(14).mean() / 
                                      df["Price"].diff(1).clip(upper=0).abs().rolling(14).mean())))
    
    df.to_csv("data/processed/kenya_processed.csv", index=False)
    return df

clean_historical_data("data/kenya/nse_historical.csv")