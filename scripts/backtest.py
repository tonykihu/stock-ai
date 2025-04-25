import pandas as pd
import joblib

# Load model and data
model = joblib.load("models/hybrid_model.pkl")
data = pd.read_csv("data/processed/features.csv")

# Simulate trades
data["Prediction"] = model.predict(data[["rsi_14", "sma_50", "Score"]])
data["Returns"] = data["Close"].pct_change().shift(-1)
data["Strategy_Returns"] = data["Returns"] * data["Prediction"]

# Compare vs buy-and-hold
print("Strategy CAGR:", (1 + data["Strategy_Returns"]).cumprod()[-1])
print("Buy-and-Hold CAGR:", (1 + data["Returns"]).cumprod()[-1])

from sklearn.model_selection import GridSearchCV

params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), params, cv=5)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)