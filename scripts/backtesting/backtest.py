import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load model and data
model = joblib.load("models/hybrid_model.pkl")
data = pd.read_csv("data/processed/features.csv")

# Merge with sentiment data for the sentiment_score column
sentiment = pd.read_csv("data/processed/news_sentiment.csv")
data = pd.merge(data, sentiment, on="Date")

# Simulate trades
data["Prediction"] = model.predict(data[["rsi_14", "sma_50", "sentiment_score"]])
data["Returns"] = data["Close"].pct_change().shift(-1)
data["Strategy_Returns"] = data["Returns"] * data["Prediction"]

# Drop NaN rows
data = data.dropna()

# Compare vs buy-and-hold
print(f"Strategy CAGR: {(1 + data['Strategy_Returns']).cumprod().iloc[-1]:.4f}")
print(f"Buy-and-Hold CAGR: {(1 + data['Returns']).cumprod().iloc[-1]:.4f}")

# Hyperparameter tuning
# Reload training data for grid search
train_data = pd.read_csv("data/processed/features.csv").dropna()
train_data = pd.merge(train_data, sentiment, on="Date").dropna()

X_train = train_data[["rsi_14", "sma_50", "sentiment_score"]]
y_train = (train_data["Close"].shift(-1) > train_data["Close"]).astype(int)

# Align after shift
X_train = X_train.iloc[:-1]
y_train = y_train.iloc[:-1]

params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
