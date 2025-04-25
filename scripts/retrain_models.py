from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Load current model and new data
model = joblib.load("models/hybrid_model.pkl")
new_data = pd.read_csv("data/processed/latest.csv")

# Check accuracy drop
old_acc = 0.72  # Baseline accuracy
new_acc = accuracy_score(new_data["Target"], model.predict(new_data[features]))
if new_acc < old_acc - 0.1:  # Significant drop
    print("Model drift detected - retraining...")
    retrain_model()