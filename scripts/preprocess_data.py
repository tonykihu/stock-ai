import pandas as pd

# Load datasets
us_data = pd.read_csv("data/us/aapl.csv", parse_dates=["Date"])
kenya_data = pd.read_csv("data/kenya/nse_latest.csv")

# Standardize columns
us_data["Market"] = "US"
kenya_data["Market"] = "Kenya"
combined = pd.concat([us_data, kenya_data], ignore_index=True)

# Save processed data
combined.to_csv("data/processed/combined.csv", index=False)

from ta import add_all_ta_features

combined = pd.read_csv("data/processed/combined.csv")

# Add indicators (for US data; adjust for Kenya if needed)
combined = add_all_ta_features(
    combined, open="Open", high="High", low="Low", 
    close="Close", volume="Volume"
)

# Keep key features
features = ["rsi_14", "macd", "sma_50", "volume_obv"]
combined[features + ["Market", "Date"]].to_csv("data/processed/features.csv", index=False)