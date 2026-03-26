"""
src/models/train.py
Trains experiments one at a time, saves each result immediately after training.
Supports --smoke-test flag for a single experiment.
"""
import os
import json
import pickle
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "models")
)
RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "tables")
)


def _get_clf(model_name: str, weighted: bool):
    """Return a fresh classifier instance."""
    cw = "balanced" if weighted else None
    clfs = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, class_weight=cw, random_state=RANDOM_STATE),
            True   # needs scaling
        ),
        "DecisionTree": (
            DecisionTreeClassifier(class_weight=cw, random_state=RANDOM_STATE),
            False
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=100, class_weight=cw,
                                   random_state=RANDOM_STATE, n_jobs=-1),
            False
        ),
        "SVM": (
            SVC(probability=True, class_weight=cw, random_state=RANDOM_STATE),
            True
        ),
        "kNN": (
            KNeighborsClassifier(n_neighbors=5),
            True
        ),
        "NaiveBayes": (
            GaussianNB(),
            False
        ),
    }
    return clfs[model_name]


def _supports_class_weight(model_name: str) -> bool:
    return model_name in {"LogisticRegression", "DecisionTree", "RandomForest", "SVM"}


def run_single_experiment(
    model_name: str,
    encoding_name: str,
    imbalance_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> dict:
    """
    Train one experiment, evaluate on test set, log result, and save model to disk.
    Returns a result dict (without large arrays).
    """
    exp_id = f"{model_name}__{encoding_name}__{imbalance_name}"
    logger.info(f"  ▶ START  {exp_id}")

    # ── Prepare training data ─────────────────────────────────────
    X_tr, y_tr = X_train.copy(), y_train.copy()

    use_weighted = (imbalance_name == "ClassWeight" and _supports_class_weight(model_name))
    clf, needs_scale = _get_clf(model_name, weighted=use_weighted)

    if imbalance_name == "SMOTE":
        logger.info(f"    Applying SMOTE (minority→{y_tr.sum()} samples)...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        logger.info(f"    SMOTE done. Training size: {len(y_tr)} ({y_tr.sum()} pos)")

    # ── Scale if needed ───────────────────────────────────────────
    scaler = None
    if needs_scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_test)
    else:
        X_te = X_test

    # ── Train ─────────────────────────────────────────────────────
    logger.info(f"    Training {model_name}...")
    clf.fit(X_tr, y_tr)

    # ── Evaluate ──────────────────────────────────────────────────
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None

    result = {
        "Model":     model_name,
        "Encoding":  encoding_name,
        "Imbalance": imbalance_name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
        "_y_pred":   y_pred.tolist(),
        "_y_prob":   y_prob.tolist() if y_prob is not None else None,
    }

    logger.info(
        f"  ✔ DONE   {exp_id} | "
        f"Acc={result['Accuracy']:.4f} Prec={result['Precision']:.4f} "
        f"Rec={result['Recall']:.4f} F1={result['F1']:.4f} "
        f"AUC={result['ROC_AUC']}"
    )

    # ── Save model immediately ────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{exp_id}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler, "feature_names": feature_names}, f)
    logger.info(f"    Saved model → {model_path}")

    # ── Append to rolling results CSV ────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    import pandas as pd
    row_df = pd.DataFrame([{k: v for k, v in result.items() if not k.startswith("_")}])
    header = not os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode="a", index=False, header=header)
    logger.info(f"    Appended result → {csv_path}")

    # Attach arrays for visualization (not saved to CSV)
    result["_fitted"] = clf
    result["_X_test"] = X_test
    result["_y_test"] = y_test
    result["_feature_names"] = feature_names
    result["_scaler"] = scaler

    return result


def get_experiment_matrix(smoke_test: bool = False):
    """Return list of (model, encoding, imbalance) tuples."""
    models = ["LogisticRegression", "DecisionTree", "RandomForest", "kNN", "NaiveBayes"]
    encodings = ["OHE", "LabelEnc"]
    imbalances = ["None", "ClassWeight", "SMOTE"]

    if smoke_test:
        # Single fast experiment: LogisticRegression + OHE + None
        return [("LogisticRegression", "OHE", "None")]

    matrix = []
    for enc in encodings:
        for imb in imbalances:
            for mdl in models:
                matrix.append((mdl, enc, imb))
    return matrix
