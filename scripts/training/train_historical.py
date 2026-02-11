# scripts/train_historical_data.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load processed data
data = pd.read_csv("data/processed/kenya_processed.csv")

# Create target: 1 if price rises next week, 0 otherwise
data["Target"] = (data["Price"].shift(-5) > data["Price"]).astype(int)

# Drop rows with NaN values (from rolling indicators and shift)
data = data.dropna(subset=["sma_50", "rsi_14", "Target"])

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(data[["sma_50", "rsi_14"]], data["Target"])

print(f"Model trained on {len(data)} samples.")

# Save model
joblib.dump(model, "models/historical_model.pkl")
print("Historical model saved to models/historical_model.pkl")
