"""
src/visualization/plots.py
Generates all required charts and saves to outputs/charts/.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                              ConfusionMatrixDisplay)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHARTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "charts")
)

PALETTE = sns.color_palette("husl", 6)
PLT_STYLE = "seaborn-v0_8-whitegrid"


def _savefig(fig, filename: str):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved chart: {path}")
    return path


def plot_roc_curves(results: list, X_test: np.ndarray, y_test: np.ndarray):
    """Plot ROC curve for each model's best OHE configuration."""
    # Pick best result per model (by ROC_AUC, OHE encoding)
    best_per_model = {}
    for r in results:
        m = r["Model"]
        if r["Encoding"] == "OHE" and r.get("ROC_AUC") and not np.isnan(r["ROC_AUC"]):
            if m not in best_per_model or r["ROC_AUC"] > best_per_model[m]["ROC_AUC"]:
                best_per_model[m] = r

    try:
        plt.style.use(PLT_STYLE)
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (model_name, r) in enumerate(best_per_model.items()):
        if r.get("_y_prob") is not None:
            fpr, tpr, _ = roc_curve(y_test, r["_y_prob"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=PALETTE[i], lw=2,
                    label=f"{model_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – All Models (Best Config, OHE)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    return _savefig(fig, "roc_curve.png")


def plot_model_performance(comparison_df: pd.DataFrame):
    """Grouped bar chart: F1 and ROC_AUC by model (best config per model)."""
    best = (
        comparison_df
        .sort_values("ROC_AUC", ascending=False)
        .groupby("Model")
        .first()
        .reset_index()
    )
    models = best["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    try:
        plt.style.use(PLT_STYLE)
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, best["F1"], width, label="F1-Score",
                    color="#4C72B0", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, best["ROC_AUC"], width, label="ROC-AUC",
                    color="#DD8452", alpha=0.85, edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison (Best Configuration per Model)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    return _savefig(fig, "model_performance.png")


def plot_feature_importance(model, feature_names: list, top_n: int = 15):
    """Horizontal bar chart of Random Forest feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    try:
        plt.style.use(PLT_STYLE)
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
    ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances – Random Forest",
                 fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    return _savefig(fig, "feature_importance.png")


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Confusion matrix heatmap for the best model."""
    cm = confusion_matrix(y_test, y_pred)
    try:
        plt.style.use(PLT_STYLE)
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["No (0)", "Yes (1)"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix – {model_name} (Best Config)",
                 fontsize=13, fontweight="bold")
    return _savefig(fig, "confusion_matrix.png")


def plot_imbalance_impact(comparison_df: pd.DataFrame):
    """Bar chart showing Recall/F1 across imbalance strategies."""
    avg = (
        comparison_df
        .groupby("Imbalance")[["Recall", "F1"]]
        .mean()
        .reset_index()
    )
    try:
        plt.style.use(PLT_STYLE)
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(avg))
    width = 0.35
    ax.bar(x - width / 2, avg["Recall"], width, label="Recall", color="#55A868", alpha=0.85)
    ax.bar(x + width / 2, avg["F1"], width, label="F1-Score", color="#C44E52", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(avg["Imbalance"])
    ax.set_ylabel("Score (averaged across models)")
    ax.set_title("Impact of Imbalance Handling on Recall & F1",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend()
    return _savefig(fig, "imbalance_impact.png")
