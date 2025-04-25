import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load technical features and sentiment
tech_data = pd.read_csv("data/processed/features.csv")
sentiment = pd.read_csv("data/processed/news_sentiment.csv")

# Merge (example: assuming same dates)
merged = pd.merge(tech_data, sentiment, on="Date")

# Train hybrid model
X = merged[["rsi_14", "sma_50", "Score"]]
y = merged["Target"]  # From earlier
model = RandomForestClassifier().fit(X, y)

# Save
joblib.dump(model, "models/hybrid_model.pkl")