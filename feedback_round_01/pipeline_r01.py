"""
feedback_round_01/pipeline_r01.py
Improved pipeline addressing all fb_r01.md feedback items.
Uses ASCII-safe logging (no emoji) to avoid Windows cp1252 encoding errors.
"""
import os, sys, json, pickle, logging, argparse, warnings
# Windows terminal encoding fix
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.download import download_dataset
from data.preprocess import preprocess
from features.engineering import get_feature_sets

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                      cross_val_score, GridSearchCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, auc,
                              precision_recall_curve, average_precision_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
R01_DIR    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(R01_DIR, "outputs")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
CHARTS_DIR = os.path.join(OUT_DIR, "charts")
MODELS_DIR = os.path.join(OUT_DIR, "models")
LOGS_DIR   = os.path.join(OUT_DIR, "logs")
for d in [TABLES_DIR, CHARTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
N_CV_FOLDS   = 5
PLT_STYLE    = "seaborn-v0_8-whitegrid"

# ── Logging ────────────────────────────────────────────────────────────────────
log_file = os.path.join(LOGS_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger("pipeline_r01")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def _make_model(name: str, weighted: bool = False):
    """Return (clf, needs_scale) for a model name."""
    cw = "balanced" if weighted else None
    registry = {
        "Baseline":          (DummyClassifier(strategy="most_frequent"), False),
        "LogisticRegression":(LogisticRegression(max_iter=1000, class_weight=cw,
                               random_state=RANDOM_STATE), True),
        "DecisionTree":      (DecisionTreeClassifier(class_weight=cw,
                               random_state=RANDOM_STATE), False),
        "RandomForest":      (RandomForestClassifier(n_estimators=100, class_weight=cw,
                               random_state=RANDOM_STATE, n_jobs=1), False),
        "kNN":               (KNeighborsClassifier(n_neighbors=5), True),
        "NaiveBayes":        (GaussianNB(), False),
    }
    return registry[name]


def _supports_class_weight(name):
    return name not in {"Baseline", "kNN", "NaiveBayes"}


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT MATRIX
# fb_r01 fix §3.5: LE only for tree models
# ══════════════════════════════════════════════════════════════════════════════

def build_matrix(smoke_test: bool = False):
    """
    Returns list of (model_name, encoding, imbalance_strategy).
    LE restricted to tree models (DT, RF) as per fb_r01 §3.5.
    """
    if smoke_test:
        return [("LogisticRegression", "OHE", "None")]

    ohe_models = ["Baseline", "LogisticRegression", "DecisionTree",
                  "RandomForest", "kNN", "NaiveBayes"]
    le_models  = ["DecisionTree", "RandomForest"]          # LE only for trees
    imbalances = ["None", "ClassWeight", "SMOTE"]

    matrix = []
    for imb in imbalances:
        for m in ohe_models:
            matrix.append((m, "OHE", imb))
        for m in le_models:
            matrix.append((m, "LE", imb))
    return matrix


# ══════════════════════════════════════════════════════════════════════════════
# CORE EXPERIMENT RUNNER
# fb_r01 fix §3.2: adds 5-fold CV with mean ± std
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(model_name, encoding, imbalance,
                   X_train, y_train, X_test, y_test, feature_names):
    exp_id = f"{model_name}__{encoding}__{imbalance}"
    logger.info(f"  >> {exp_id}")

    X_tr, y_tr = X_train.copy(), y_train.copy()
    weighted   = (imbalance == "ClassWeight" and _supports_class_weight(model_name))
    clf, needs_scale = _make_model(model_name, weighted=weighted)

    # SMOTE
    if imbalance == "SMOTE" and model_name != "Baseline":
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        logger.info(f"    SMOTE done. Training size: {len(y_tr)} ({y_tr.sum()} pos)")

    # Scale
    scaler = None
    if needs_scale:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_test)
    else:
        X_tr_s, X_te_s = X_tr, X_test

    # ── 5-fold CV on training data (fb_r01 §3.2) ──────────────────
    # Note: CV is done on original (unbalanced / unSMOTEd) X_train for fair comparison
    clf_cv, needs_scale_cv = _make_model(model_name, weighted=weighted)
    cv_X = scaler.fit_transform(X_train) if needs_scale else X_train
    cv    = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_f1  = cross_val_score(clf_cv, cv_X, y_train, cv=cv,
                              scoring="f1", n_jobs=1)
    cv_auc = cross_val_score(clf_cv, cv_X, y_train, cv=cv,
                              scoring="roc_auc", n_jobs=1)
    cv_rec = cross_val_score(clf_cv, cv_X, y_train, cv=cv,
                              scoring="recall", n_jobs=1)

    # ── Train final model ──────────────────────────────────────────
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)
    y_prob = clf.predict_proba(X_te_s)[:, 1] if hasattr(clf, "predict_proba") else None

    # Test-set metrics
    def safe(fn, *a, **kw):
        try:    return round(fn(*a, **kw), 4)
        except: return None

    result = {
        "Model":     model_name,
        "Encoding":  encoding,
        "Imbalance": imbalance,
        "Accuracy":  safe(accuracy_score, y_test, y_pred),
        "Precision": safe(precision_score, y_test, y_pred, zero_division=0),
        "Recall":    safe(recall_score,    y_test, y_pred, zero_division=0),
        "F1":        safe(f1_score,        y_test, y_pred, zero_division=0),
        "ROC_AUC":   safe(roc_auc_score,   y_test, y_prob) if y_prob is not None else None,
        # CV metrics
        "CV_F1_mean":  round(cv_f1.mean(),  4),
        "CV_F1_std":   round(cv_f1.std(),   4),
        "CV_AUC_mean": round(cv_auc.mean(), 4),
        "CV_AUC_std":  round(cv_auc.std(),  4),
        "CV_Rec_mean": round(cv_rec.mean(), 4),
        "CV_Rec_std":  round(cv_rec.std(),  4),
        # CV fold arrays for Wilcoxon
        "_cv_f1_folds":  cv_f1.tolist(),
        "_cv_auc_folds": cv_auc.tolist(),
        # Inference artifacts
        "_fitted":       clf,
        "_scaler":       scaler,
        "_y_pred":       y_pred,
        "_y_prob":       y_prob,
        "_y_test":       y_test,
        "_X_test":       X_te_s,
        "_feature_names":feature_names,
    }

    logger.info(
        f"  OK {exp_id}  "
        f"Acc={result['Accuracy']:.4f} Prec={result['Precision']:.4f} "
        f"Rec={result['Recall']:.4f} F1={result['F1']:.4f} "
        f"AUC={result['ROC_AUC']} | CV_F1={result['CV_F1_mean']}+/-{result['CV_F1_std']}"
    )

    # ── Save model immediately (fb_r01 implementation) ────────────
    pkl_path = os.path.join(MODELS_DIR, f"{exp_id}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler,
                     "feature_names": feature_names}, f)

    # ── Append row to CSV immediately ──────────────────────────────
    csv_path = os.path.join(TABLES_DIR, "model_comparison_r01.csv")
    row_cols  = {k: v for k, v in result.items() if not k.startswith("_")}
    row_df    = pd.DataFrame([row_cols])
    # Only write header if file is truly missing (NOT just empty or has duplicate headers)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    row_df.to_csv(csv_path, mode="a", index=False, header=not file_exists)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# RF HYPERPARAMETER TUNING  (fb_r01 §4.2)
# ══════════════════════════════════════════════════════════════════════════════

def tune_random_forest(X_train, y_train, needs_scale=False):
    """Manual RF hyperparameter search (avoids numpy GridSearchCV StrDType bug on Windows)."""
    logger.info("  [RF Tuning] Manual grid search starting...")
    param_grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": m}
        for n in [100, 200]
        for d in [None, 10]
        for m in [1, 5]
    ]
    scaler = StandardScaler() if needs_scale else None
    X_s = scaler.fit_transform(X_train) if scaler else X_train

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_score, best_params, best_model = -1, None, None

    for params in param_grid:
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)
        fold_aucs = []
        for tr_idx, val_idx in cv.split(X_s, y_train):
            clf.fit(X_s[tr_idx], y_train[tr_idx])
            prob = clf.predict_proba(X_s[val_idx])[:, 1]
            try:
                fold_aucs.append(roc_auc_score(y_train[val_idx], prob))
            except Exception:
                fold_aucs.append(0.0)
        mean_auc = float(np.mean(fold_aucs))
        logger.info(f"    params={params}  CV_AUC={mean_auc:.4f}")
        if mean_auc > best_score:
            best_score = mean_auc
            best_params = params
            best_model = clf

    logger.info(f"  [RF Tuning] Best params: {best_params}  AUC={best_score:.4f}")
    final = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **best_params)
    return final, best_params, scaler


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE  (fb_r01 §3.3)
# ══════════════════════════════════════════════════════════════════════════════

def wilcoxon_significance(results: list, metric: str = "_cv_f1_folds", top_n: int = 3):
    """
    Run Wilcoxon signed-rank test between top-N models (by mean CV F1).
    Returns DataFrame of pairwise p-values.
    """
    ohe = [r for r in results if r["Encoding"] == "OHE" and r.get("CV_F1_mean") is not None]
    ohe_sorted = sorted(ohe, key=lambda r: r["CV_F1_mean"], reverse=True)

    # Deduplicate by model name (take best config per model)
    seen, top = set(), []
    for r in ohe_sorted:
        if r["Model"] not in seen:
            seen.add(r["Model"])
            top.append(r)
        if len(top) >= top_n:
            break

    records = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            folds_a = np.array(a[metric])
            folds_b = np.array(b[metric])
            try:
                stat, pval = stats.wilcoxon(folds_a, folds_b)
            except Exception:
                stat, pval = np.nan, np.nan
            records.append({
                "Model_A":   a["Model"],
                "Imb_A":     a["Imbalance"],
                "Model_B":   b["Model"],
                "Imb_B":     b["Imbalance"],
                "Wilcoxon_stat": round(stat, 3) if not np.isnan(stat) else "—",
                "p_value":   round(pval, 4) if not np.isnan(pval) else "—",
                "Significant (p<0.05)": "Yes" if (not np.isnan(pval) and pval < 0.05) else "No",
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(TABLES_DIR, "wilcoxon_test.csv"), index=False)
    logger.info(f"  Wilcoxon test saved. Pairs: {len(records)}")
    return df, top


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Chart saved: {name}")
    return path


def plot_roc_curves(results):
    """ROC curves for best OHE config per model."""
    best = {}
    for r in results:
        if r["Encoding"] == "OHE" and r.get("ROC_AUC"):
            m = r["Model"]
            if m not in best or r["ROC_AUC"] > best[m]["ROC_AUC"]:
                best[m] = r

    try: plt.style.use(PLT_STYLE)
    except: pass
    palette = sns.color_palette("husl", len(best))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, r) in enumerate(best.items()):
        if r.get("_y_prob") is not None:
            fpr, tpr, _ = roc_curve(r["_y_test"], r["_y_prob"])
            label = f"{name} (AUC={r['ROC_AUC']:.3f})"
            ls = "--" if name == "Baseline" else "-"
            ax.plot(fpr, tpr, color=palette[i], lw=2, ls=ls, label=label)
    ax.plot([0,1],[0,1], "gray", ls=":", lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curves – All Models (Best OHE Config)",
           xlim=[0,1], ylim=[0,1.02])
    ax.legend(loc="lower right", fontsize=9)
    return _savefig(fig, "roc_curve_r01.png")


def plot_pr_curves(results):
    """Precision-Recall curves (better for imbalanced data — fb_r01 §4.3)."""
    best = {}
    for r in results:
        if r["Encoding"] == "OHE" and r.get("_y_prob") is not None:
            m = r["Model"]
            if m not in best or (r.get("ROC_AUC") or 0) > (best[m].get("ROC_AUC") or 0):
                best[m] = r

    try: plt.style.use(PLT_STYLE)
    except: pass
    palette = sns.color_palette("husl", len(best))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, r) in enumerate(best.items()):
        prec, rec, _ = precision_recall_curve(r["_y_test"], r["_y_prob"])
        ap = average_precision_score(r["_y_test"], r["_y_prob"])
        ls = "--" if name == "Baseline" else "-"
        ax.plot(rec, prec, color=palette[i], lw=2, ls=ls,
                label=f"{name} (AP={ap:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curves (Best OHE Config per Model)",
           xlim=[0,1], ylim=[0,1.02])
    ax.legend(loc="upper right", fontsize=9)
    return _savefig(fig, "pr_curve_r01.png")


def plot_cv_comparison(results):
    """CV mean±std bar chart (F1) per model — fb_r01 §3.2."""
    ohe = [r for r in results if r["Encoding"] == "OHE" and r.get("CV_F1_mean") is not None]
    best = {}
    for r in ohe:
        m = r["Model"]
        if m not in best or r["CV_F1_mean"] > best[m]["CV_F1_mean"]:
            best[m] = r

    df = pd.DataFrame(best.values()).sort_values("CV_F1_mean", ascending=False)
    models = df["Model"].tolist()
    means  = df["CV_F1_mean"].values
    stds   = df["CV_F1_std"].values

    try: plt.style.use(PLT_STYLE)
    except: pass
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#cccccc" if m == "Baseline" else "#4C72B0" for m in models]
    bars = ax.bar(models, means, yerr=stds, capsize=5,
                   color=colors, edgecolor="white", alpha=0.9)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.005,
                f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("CV F1-Score (mean ± std)")
    ax.set_title("5-Fold CV F1-Score per Model (Best OHE Config)\n"
                 "Grey = Baseline (majority class)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=15, ha="right")
    return _savefig(fig, "cv_f1_comparison_r01.png")


def plot_model_performance(results):
    """Test-set F1 & ROC-AUC grouped bar chart."""
    ohe = [r for r in results if r["Encoding"] == "OHE"]
    best = {}
    for r in ohe:
        m = r["Model"]
        if m not in best or (r.get("ROC_AUC") or 0) > (best[m].get("ROC_AUC") or 0):
            best[m] = r

    df = pd.DataFrame(best.values()).sort_values("ROC_AUC", ascending=False)
    models = df["Model"].tolist()
    x = np.arange(len(models))
    w = 0.35

    try: plt.style.use(PLT_STYLE)
    except: pass
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w/2, df["F1"].values, w, label="F1",
                 color="#4C72B0", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, df["ROC_AUC"].values, w, label="ROC-AUC",
                 color="#DD8452", alpha=0.85, edgecolor="white")
    for b in list(b1) + list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set(ylabel="Score", title="Test-Set Performance: F1 & ROC-AUC by Model",
           ylim=[0, 1.1], xticks=x, xticklabels=models)
    plt.xticks(rotation=15, ha="right")
    ax.legend(fontsize=11)
    return _savefig(fig, "model_performance_r01.png")


def plot_feature_importance(model, feature_names, top_n=15):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    try: plt.style.use(PLT_STYLE)
    except: pass
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
    ax.barh(range(top_n), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Feature Importances – Random Forest (Best Config)",
                 fontsize=12, fontweight="bold")
    return _savefig(fig, "feature_importance_r01.png")


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    try: plt.style.use(PLT_STYLE)
    except: pass
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=12, fontweight="bold")
    return _savefig(fig, "confusion_matrix_r01.png")


def plot_imbalance_impact(results):
    ohe = [r for r in results
           if r["Encoding"] == "OHE" and r["Model"] != "Baseline"]
    avg = (pd.DataFrame(ohe)
             .groupby("Imbalance")[["Recall", "F1"]]
             .mean().reset_index())

    try: plt.style.use(PLT_STYLE)
    except: pass
    x = np.arange(len(avg))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, avg["Recall"], w, label="Recall",  color="#55A868", alpha=0.85)
    ax.bar(x + w/2, avg["F1"],     w, label="F1-Score", color="#C44E52", alpha=0.85)
    ax.set(xticks=x, xticklabels=avg["Imbalance"], ylim=[0, 1],
           ylabel="Score (mean across models)",
           title="Impact of Imbalance Handling on Recall & F1\n(OHE, averaged over non-baseline models)",)
    ax.legend()
    return _savefig(fig, "imbalance_impact_r01.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(smoke_test=False):
    logger.info("=" * 65)
    logger.info("Pipeline R01 -- Improved (feedback_round_01)")
    mode_str = "SMOKE TEST" if smoke_test else "FULL"
    logger.info(f"Mode: {mode_str}")
    logger.info("=" * 65)

    # Clean stale CSV
    csv_path = os.path.join(TABLES_DIR, "model_comparison_r01.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
        logger.info("Cleared stale model_comparison_r01.csv")
    # ── Data ────────────────────────────────────────────────────
    logger.info("\n[1/6] Download + Preprocess")
    raw_csv = download_dataset()
    data    = preprocess(raw_csv, drop_leakage=True)
    X_ohe, X_le = data["X_ohe"], data["X_le"]
    y            = data["y"]
    fn_ohe, fn_le = data["feature_names_ohe"], data["feature_names_le"]
    logger.info(f"  Rows={len(y)}  Positive={y.mean():.2%}")

    # ── Feature sets ─────────────────────────────────────────────
    logger.info("\n[2/6] Feature engineering")
    feat_sets = get_feature_sets(X_ohe, y, fn_ohe, X_le, fn_le)

    # ── Split ────────────────────────────────────────────────────
    logger.info("\n[3/6] Train/test split (80/20, stratified)")
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2,
                                       random_state=RANDOM_STATE, stratify=y)
    enc_data = {
        "OHE": (feat_sets["full_ohe"][0], fn_ohe),
        "LE":  (feat_sets["full_le"][0],  fn_le),
    }

    # ── Experiments ──────────────────────────────────────────────
    matrix = build_matrix(smoke_test=smoke_test)
    logger.info(f"\n[4/6] Running {len(matrix)} experiments...\n")
    all_results = []

    for i, (model_name, enc, imb) in enumerate(matrix, 1):
        logger.info(f"-- [{i}/{len(matrix)}] {model_name} | {enc} | {imb}")
        X, feat_names = enc_data[enc]
        try:
            r = run_experiment(
                model_name, enc, imb,
                X[idx_tr], y[idx_tr],
                X[idx_te], y[idx_te],
                feat_names,
            )
            all_results.append(r)
        except Exception as e:
            logger.info(f"  FAIL: {e}")

    # ── RF Hyperparameter Tuning ──────────────────────────────────
    if not smoke_test:
        logger.info("\n[5/6] RF Hyperparameter Tuning (OHE)")
        X_ohe_tr = feat_sets["full_ohe"][0][idx_tr]
        best_rf, best_params, _ = tune_random_forest(X_ohe_tr, y[idx_tr])

        # Re-evaluate tuned RF on test set
        X_ohe_te = feat_sets["full_ohe"][0][idx_te]
        best_rf.fit(X_ohe_tr, y[idx_tr])
        yp = best_rf.predict(X_ohe_te)
        yprob = best_rf.predict_proba(X_ohe_te)[:, 1]
        tuned_row = {
            "Model": "RandomForest_Tuned", "Encoding": "OHE", "Imbalance": "Tuned",
            "Accuracy":  round(accuracy_score(y[idx_te], yp), 4),
            "Precision": round(precision_score(y[idx_te], yp, zero_division=0), 4),
            "Recall":    round(recall_score(y[idx_te], yp, zero_division=0), 4),
            "F1":        round(f1_score(y[idx_te], yp, zero_division=0), 4),
            "ROC_AUC":   round(roc_auc_score(y[idx_te], yprob), 4),
            "CV_F1_mean": "—", "CV_F1_std": "—",
            "CV_AUC_mean": "—", "CV_AUC_std": "—",
            "CV_Rec_mean": "—", "CV_Rec_std": "—",
            "_fitted": best_rf, "_y_pred": yp, "_y_prob": yprob,
            "_y_test": y[idx_te], "_X_test": X_ohe_te,
            "_feature_names": fn_ohe, "_cv_f1_folds": [], "_cv_auc_folds": [],
        }
        all_results.append(tuned_row)
        logger.info(f"  Tuned RF -> F1={tuned_row['F1']}  AUC={tuned_row['ROC_AUC']}")
        logger.info(f"  Best params: {best_params}")

        # Save tuned model
        pkl = os.path.join(MODELS_DIR, "RandomForest_Tuned__OHE__Tuned.pkl")
        with open(pkl, "wb") as f:
            pickle.dump({"model": best_rf, "scaler": None,
                         "best_params": best_params, "feature_names": fn_ohe}, f)

        # Append to CSV
        row_df = pd.DataFrame([{k: v for k, v in tuned_row.items()
                                 if not k.startswith("_")}])
        row_df.to_csv(csv_path, mode="a", index=False, header=False)

    # ── Statistical Significance ──────────────────────────────────
    wilcoxon_df = None
    if not smoke_test and len(all_results) >= 3:
        logger.info("\n[5b/6] Wilcoxon significance test")
        wilcoxon_df, top_models = wilcoxon_significance(all_results, top_n=3)
        logger.info(f"\n{wilcoxon_df.to_string(index=False)}")

    # ── Charts ────────────────────────────────────────────────────
    logger.info("\n[6/6] Generating charts")
    ohe_results = [r for r in all_results if r["Encoding"] == "OHE"]
    y_te = y[idx_te]

    plot_roc_curves(all_results)
    plot_pr_curves(all_results)
    plot_cv_comparison(all_results)
    plot_model_performance(all_results)
    plot_imbalance_impact(all_results)

    # Feature importance from best RF (non-tuned, for transparency)
    rf_res = [r for r in all_results
              if r["Model"] == "RandomForest" and r["Encoding"] == "OHE"
              and r.get("_fitted") and hasattr(r["_fitted"], "feature_importances_")]
    if rf_res:
        best_rf_r = max(rf_res, key=lambda r: r.get("ROC_AUC") or 0)
        plot_feature_importance(best_rf_r["_fitted"], best_rf_r["_feature_names"])

    # Confusion matrix of best model
    valid = [r for r in all_results if r.get("ROC_AUC") and r.get("_y_pred") is not None
             and r["Model"] != "Baseline"]
    if valid:
        best = max(valid, key=lambda r: r["ROC_AUC"])
        plot_confusion_matrix(best["_y_test"], best["_y_pred"],
                               f"{best['Model']} ({best['Encoding']}, {best['Imbalance']})")

    # ── Save metadata for abstract ────────────────────────────────
    valid_meta = [r for r in all_results if r.get("ROC_AUC")
                  and r["Model"] not in {"Baseline", "RandomForest_Tuned"}]
    best_m = max(valid_meta, key=lambda r: r["ROC_AUC"])
    baseline_r = next((r for r in all_results if r["Model"] == "Baseline"
                        and r["Encoding"] == "OHE" and r["Imbalance"] == "None"), {})

    top_feats = []
    if rf_res:
        br = max(rf_res, key=lambda r: r.get("ROC_AUC") or 0)
        sorted_idx = np.argsort(br["_fitted"].feature_importances_)[::-1][:10]
        top_feats = [br["_feature_names"][i] for i in sorted_idx]

    metadata = {
        "best_model": {k: v for k, v in best_m.items() if not k.startswith("_")},
        "baseline":   {k: v for k, v in baseline_r.items() if not k.startswith("_")},
        "dataset_shape": [int(X_ohe.shape[0]), int(X_ohe.shape[1])],
        "positive_rate": float(y.mean()),
        "n_experiments": len(all_results),
        "top_features":  top_feats,
        "wilcoxon": wilcoxon_df.to_dict(orient="records") if wilcoxon_df is not None else [],
    }
    with open(os.path.join(OUT_DIR, "metadata_r01.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata saved.")

    logger.info("\n" + "=" * 65)
    logger.info(f"R01 Pipeline complete -- {len(all_results)} experiments")
    logger.info(f"Best: {best_m['Model']} | {best_m['Encoding']} | "
                f"{best_m['Imbalance']} | AUC={best_m['ROC_AUC']} F1={best_m['F1']}")
    logger.info("=" * 65)
    return all_results, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
