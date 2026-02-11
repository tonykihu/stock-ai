import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load technical features and sentiment
tech_data = pd.read_csv("data/processed/features.csv")
sentiment = pd.read_csv("data/processed/news_sentiment.csv")

# Average sentiment per date (in case multiple headlines per day)
sentiment_daily = sentiment.groupby("Date", as_index=False)["sentiment_score"].mean()

# Left merge so all feature rows are kept; fill missing sentiment with neutral 0.5
merged = pd.merge(tech_data, sentiment_daily, on="Date", how="left")
merged["sentiment_score"] = merged["sentiment_score"].fillna(0.5)

# Create target: 1 if price rises next day, else 0
merged["Target"] = (merged["Close"].shift(-1) > merged["Close"]).astype(int)

# Drop rows with NaN (from shift and missing technical indicators during warmup)
merged = merged.dropna()

# Train hybrid model
X = merged[["rsi_14", "sma_50", "sentiment_score"]]
y = merged["Target"]
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

# Save
joblib.dump(model, "models/hybrid_model.pkl")
print(f"Hybrid model trained on {len(X)} samples and saved.")
