import os
import sys
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.features import HYBRID_FEATURES, TECHNICAL_FEATURES
from utils.evaluation import walk_forward_validate, save_metrics

# Load technical features and sentiment
tech_data = pd.read_csv("data/processed/features.csv")
try:
    sentiment = pd.read_csv("data/processed/news_sentiment.csv")
    # Average sentiment per date+ticker (or just date if no Ticker column)
    group_cols = ["Date", "Ticker"] if "Ticker" in sentiment.columns else ["Date"]
    sentiment_daily = sentiment.groupby(group_cols, as_index=False)["sentiment_score"].mean()
    # Merge on available keys
    merge_cols = [c for c in group_cols if c in tech_data.columns]
    merged = pd.merge(tech_data, sentiment_daily, on=merge_cols, how="left")
except FileNotFoundError:
    print("No sentiment data found — using neutral 0.5 for all rows")
    merged = tech_data.copy()

merged["sentiment_score"] = merged["sentiment_score"].fillna(0.5) if "sentiment_score" in merged.columns else 0.5

# Train on AAPL
merged = merged[merged["Ticker"] == "AAPL"].copy()
merged = merged.sort_values("Date").reset_index(drop=True)

# Create target: 1 if price rises next day, else 0
merged["Target"] = (merged["Close"].shift(-1) > merged["Close"]).astype(int)
merged = merged.dropna(subset=HYBRID_FEATURES + ["Target"])

X = merged[HYBRID_FEATURES]
y = merged["Target"]

print(f"Training data: {len(X)} samples, {len(HYBRID_FEATURES)} features")
print(f"Class balance: UP={int(y.sum())}/{len(y)} ({y.mean():.1%})")

# Walk-forward cross-validation
model_params = {"n_estimators": 100, "random_state": 42}
cv_results = walk_forward_validate(
    RandomForestClassifier, model_params, X, y, n_splits=5
)

print("\nWalk-Forward CV Results:")
for fold in cv_results["folds"]:
    auc_str = f", AUC={fold['roc_auc']:.4f}" if fold.get("roc_auc") else ""
    print(f"  Fold {fold['fold']}: acc={fold['accuracy']:.4f}, "
          f"f1={fold['f1']:.4f}, prec={fold['precision']:.4f}, "
          f"rec={fold['recall']:.4f}{auc_str} "
          f"(train={fold['train_size']}, test={fold['test_size']})")

agg = cv_results["aggregate"]
print(f"\n  Mean Accuracy:  {agg['mean_accuracy']:.4f} (+/- {agg['std_accuracy']:.4f})")
print(f"  Mean F1:        {agg['mean_f1']:.4f}")
if "mean_roc_auc" in agg:
    print(f"  Mean AUC:       {agg['mean_roc_auc']:.4f}")

# Train final model on all data
model = RandomForestClassifier(**model_params).fit(X, y)

# Save
save_metrics(cv_results, "hybrid")
joblib.dump(model, "models/hybrid_model.pkl")
print(f"\nHybrid model trained on {len(X)} samples and saved.")
