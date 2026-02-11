import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load features
data = pd.read_csv("data/processed/features.csv").dropna()

# Create target: 1 if price rises next day, else 0
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

# Drop the last row which has NaN target from shift
data = data.iloc[:-1]

# Split data
X = data[["rsi_14", "sma_50", "volume_obv"]]
y = data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")

# Save model
joblib.dump(model, "models/technical_model.pkl")
print("Technical model saved to models/technical_model.pkl")
