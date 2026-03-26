"""
run_pipeline.py
Main entrypoint. Runs experiments one-at-a-time with real-time logging.

Usage:
    python run_pipeline.py               # full 36 experiments
    python run_pipeline.py --smoke-test  # 1 experiment (fast verification)
"""
import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.download import download_dataset
from data.preprocess import preprocess
from features.engineering import get_feature_sets
from models.train import run_single_experiment, get_experiment_matrix
from evaluation.evaluate import get_best_result
from visualization.plots import (plot_roc_curves, plot_model_performance,
                                  plot_feature_importance, plot_confusion_matrix,
                                  plot_imbalance_impact)

# ── Logging setup ──────────────────────────────────────────────────────────────
os.makedirs("outputs/logs", exist_ok=True)
log_file = f"outputs/logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),          # real-time console output
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger("pipeline")

RANDOM_STATE = 42


def main(smoke_test: bool = False):
    logger.info("=" * 60)
    logger.info("Bank Marketing ML Pipeline")
    logger.info(f"Mode: {'SMOKE TEST (1 experiment)' if smoke_test else 'FULL (36 experiments)'}")
    logger.info("=" * 60)

    # Clean previous CSV so we don't append to stale data
    csv_path = os.path.join("outputs", "tables", "model_comparison.csv")
    if os.path.exists(csv_path) and not smoke_test:
        os.remove(csv_path)
        logger.info("Cleared previous model_comparison.csv")

    # ── Step 1: Download ───────────────────────────────────────────
    logger.info("\n[Step 1/5] Downloading dataset...")
    csv_raw = download_dataset()

    # ── Step 2: Preprocess ────────────────────────────────────────
    logger.info("\n[Step 2/5] Preprocessing...")
    data = preprocess(csv_raw, drop_leakage=True)
    X_ohe, X_le = data["X_ohe"], data["X_le"]
    y             = data["y"]
    fn_ohe, fn_le = data["feature_names_ohe"], data["feature_names_le"]
    logger.info(f"  Dataset shape: {X_ohe.shape[0]} rows | "
                f"Positive rate: {y.mean():.2%} ({y.sum()} positives)")

    # ── Step 3: Feature engineering ───────────────────────────────
    logger.info("\n[Step 3/5] Feature engineering...")
    feat_sets = get_feature_sets(X_ohe, y, fn_ohe, X_le, fn_le)

    # ── Step 4: Train/test split ──────────────────────────────────
    logger.info("\n[Step 4/5] Train/test split (80/20, stratified, seed=42)...")
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"  Train: {len(idx_train)} | Test: {len(idx_test)}")

    encoding_data = {
        "OHE":      (feat_sets["full_ohe"][0], fn_ohe),
        "LabelEnc": (feat_sets["full_le"][0], fn_le),
    }

    # ── Step 5: Run experiments ───────────────────────────────────
    matrix = get_experiment_matrix(smoke_test=smoke_test)
    total  = len(matrix)
    logger.info(f"\n[Step 5/5] Running {total} experiment(s)...\n")

    all_results = []

    for i, (model_name, enc_name, imb_name) in enumerate(matrix, 1):
        logger.info(f"─── Experiment {i}/{total}: "
                    f"{model_name} | {enc_name} | {imb_name} ───")

        X, feat_names = encoding_data[enc_name]
        X_train = X[idx_train]
        X_test  = X[idx_test]
        y_train = y[idx_train]
        y_test  = y[idx_test]

        try:
            result = run_single_experiment(
                model_name=model_name,
                encoding_name=enc_name,
                imbalance_name=imb_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=feat_names,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"  ✗ FAILED {model_name}/{enc_name}/{imb_name}: {e}")

    # ── Visualization (skip in smoke-test if needed) ───────────────
    if all_results:
        logger.info("\n[Visualization] Generating charts...")

        comp_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

        ohe_results = [r for r in all_results if r["Encoding"] == "OHE"]
        y_test_arr  = y[idx_test]

        if ohe_results:
            plot_roc_curves(ohe_results, encoding_data["OHE"][0][idx_test], y_test_arr)

        if comp_df is not None and len(comp_df) > 1:
            plot_model_performance(comp_df)
            plot_imbalance_impact(comp_df)
        else:
            logger.info("  (skipping multi-model charts — only 1 result so far)")

        # RF feature importance
        rf_results = [
            r for r in all_results
            if r["Model"] == "RandomForest" and r.get("_fitted")
        ]
        if rf_results:
            best_rf = max(rf_results, key=lambda r: r.get("ROC_AUC") or 0)
            plot_feature_importance(best_rf["_fitted"], best_rf["_feature_names"])

        # Confusion matrix for best result
        best = get_best_result(all_results, metric="ROC_AUC")
        if best and best.get("_y_pred") is not None:
            plot_confusion_matrix(
                best["_y_test"], best["_y_pred"],
                f"{best['Model']} ({best['Imbalance']})"
            )

    # ── Save metadata ──────────────────────────────────────────────
    if all_results:
        best = get_best_result(all_results, metric="ROC_AUC")
        metadata = {
            "best_model": {k: v for k, v in best.items()
                           if not k.startswith("_")},
            "dataset_shape": [int(X_ohe.shape[0]), int(X_ohe.shape[1])],
            "positive_rate": float(y.mean()),
            "n_experiments": len(all_results),
            "top_features": [],
        }
        rf_list = [r for r in all_results if r["Model"] == "RandomForest"
                   and r.get("_fitted")]
        if rf_list:
            rf = max(rf_list, key=lambda r: r.get("ROC_AUC") or 0)
            sorted_idx = np.argsort(rf["_fitted"].feature_importances_)[::-1][:10]
            metadata["top_features"] = [rf["_feature_names"][i] for i in sorted_idx]

        with open("outputs/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"Pipeline complete! {len(all_results)}/{total} experiments succeeded.")
    logger.info(f"Outputs: outputs/tables/ | outputs/charts/ | outputs/models/")
    logger.info("=" * 60)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only 1 experiment to verify the pipeline")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
