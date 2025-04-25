# scripts/train_historical.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load processed data
data = pd.read_csv("data/processed/kenya_processed.csv")

# Create target: 1 if price rises next week, 0 otherwise
data["Target"] = (data["Price"].shift(-5) > data["Price"]).astype(int)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(data[["sma_50", "rsi_14"]], data["Target"])

# Save model
import joblib
joblib.dump(model, "models/historical_model.pkl")