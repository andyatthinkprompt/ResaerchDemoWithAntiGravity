"""
src/evaluation/evaluate.py
Collects metrics, runs cross-validation, and saves model_comparison.csv.
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, make_scorer)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs"))


def build_comparison_df(results: list) -> pd.DataFrame:
    """Convert list of result dicts to a clean comparison DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "Model": r["Model"],
            "Encoding": r["Encoding"],
            "Imbalance": r["Imbalance"],
            "Accuracy": r["Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1": r["F1"],
            "ROC_AUC": r["ROC_AUC"],
        })
    return pd.DataFrame(rows)


def save_comparison(df: pd.DataFrame, path: str = None):
    """Save model comparison table to CSV."""
    if path is None:
        path = os.path.join(OUTPUTS_DIR, "tables", "model_comparison.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved model comparison to {path}")
    return path


def get_best_result(results: list, metric: str = "ROC_AUC") -> dict:
    """Return the result dict with the highest metric value."""
    valid = [r for r in results if r.get(metric) and not np.isnan(r[metric])]
    return max(valid, key=lambda r: r[metric])


def cross_validate_model(model_factory, X, y, n_splits: int = 5):
    """Run stratified k-fold CV and return mean metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    scores = cross_validate(model_factory, X, y, cv=cv, scoring=scoring,
                             return_train_score=False, n_jobs=-1)
    return {k.replace("test_", ""): np.mean(v) for k, v in scores.items() if k.startswith("test_")}
