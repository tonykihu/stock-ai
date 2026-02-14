"""
Time-series cross-validation and metrics reporting.

Walk-forward validation ensures no data leakage: the training set
always precedes the test set chronologically.
"""
import os
import json
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def walk_forward_validate(model_class, model_params, X, y, n_splits=5):
    """
    Walk-forward (expanding window) cross-validation.

    TimeSeriesSplit creates chronological splits:
      Fold 1: train=[0..N],     test=[N..2N]
      Fold 2: train=[0..2N],    test=[2N..3N]
      ...
    The training set always grows. The test set is always in the future.

    Args:
        model_class: e.g. RandomForestClassifier
        model_params: dict of hyperparameters
        X: feature DataFrame (must be time-sorted)
        y: target Series (must be time-sorted)
        n_splits: number of CV folds

    Returns:
        dict with per-fold metrics and aggregate metrics
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        fold_metrics = {
            "fold": fold_idx + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        if y_proba is not None:
            try:
                fold_metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                fold_metrics["roc_auc"] = None

        fold_results.append(fold_metrics)

    # Aggregate across folds
    aggregate = {
        "mean_accuracy": float(np.mean([f["accuracy"] for f in fold_results])),
        "mean_precision": float(np.mean([f["precision"] for f in fold_results])),
        "mean_recall": float(np.mean([f["recall"] for f in fold_results])),
        "mean_f1": float(np.mean([f["f1"] for f in fold_results])),
        "std_accuracy": float(np.std([f["accuracy"] for f in fold_results])),
    }

    auc_values = [f["roc_auc"] for f in fold_results if f.get("roc_auc") is not None]
    if auc_values:
        aggregate["mean_roc_auc"] = float(np.mean(auc_values))

    return {"folds": fold_results, "aggregate": aggregate}


def print_evaluation_report(y_true, y_pred, label="Model"):
    """Print classification report and confusion matrix."""
    print(f"\n{'=' * 50}")
    print(f"  {label} Evaluation Report")
    print(f"{'=' * 50}")
    print(classification_report(y_true, y_pred, target_names=["DOWN", "UP"]))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  Predicted:  DOWN  UP")
    print(f"  Actual DOWN: {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"  Actual UP:   {cm[1][0]:4d}  {cm[1][1]:4d}")
    print(f"{'=' * 50}\n")


def save_metrics(metrics, model_name, output_dir="logs"):
    """Save metrics to JSON for monitoring dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {path}")
