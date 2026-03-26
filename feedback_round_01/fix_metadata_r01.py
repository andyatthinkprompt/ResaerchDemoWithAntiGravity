"""
feedback_round_01/fix_metadata_r01.py
Reads the existing model_comparison_r01.csv and regenerates metadata_r01.json
and the research abstract without re-running experiments.
Run after pipeline_r01.py produced model_comparison_r01.csv.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
from scipy import stats

R01_DIR    = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(R01_DIR, "outputs", "tables")
MODELS_DIR = os.path.join(R01_DIR, "outputs", "models")
OUT_DIR    = os.path.join(R01_DIR, "outputs")

sys.path.insert(0, os.path.join(R01_DIR, "..", "src"))

csv_path = os.path.join(TABLES_DIR, "model_comparison_r01.csv")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from model_comparison_r01.csv")
print(df[["Model","Encoding","Imbalance","F1","ROC_AUC","CV_F1_mean","CV_F1_std"]].to_string(index=False))

# Best model (exclude baseline and tuned, pick by ROC_AUC)
valid = df[(df["Model"] != "Baseline") & (df["Model"] != "RandomForest_Tuned")]
best_row = valid.loc[valid["ROC_AUC"].idxmax()].to_dict()
baseline_row = df[(df["Model"] == "Baseline") & (df["Encoding"] == "OHE") & (df["Imbalance"] == "None")]
bl = baseline_row.iloc[0].to_dict() if len(baseline_row) > 0 else {}

print(f"\nBest: {best_row['Model']} | {best_row['Encoding']} | {best_row['Imbalance']}")
print(f"  ROC_AUC={best_row['ROC_AUC']}  F1={best_row['F1']}  Recall={best_row['Recall']}")
print(f"  CV_F1={best_row.get('CV_F1_mean','?')} ± {best_row.get('CV_F1_std','?')}")

# Load best RF model for feature importances
top_feats = []
rf_pkl = os.path.join(MODELS_DIR, "RandomForest__OHE__SMOTE.pkl")
if os.path.exists(rf_pkl):
    with open(rf_pkl, "rb") as f:
        rf_data = pickle.load(f)
    rf_model = rf_data["model"]
    rf_names = rf_data["feature_names"]
    sorted_idx = np.argsort(rf_model.feature_importances_)[::-1][:10]
    top_feats = [rf_names[i] for i in sorted_idx]
    print(f"\nTop features: {top_feats[:5]}")

# Wilcoxon on CV fold scores stored per model
# (folds not in CSV — use CV_F1_mean as a proxy for significance table)
# Attempt Wilcoxon from stored metadata if available
wilcoxon = []

metadata = {
    "best_model":    {k: (None if pd.isna(v) else v) for k, v in best_row.items()},
    "baseline":      {k: (None if pd.isna(v) else v) for k, v in bl.items()},
    "dataset_shape": [41188, 45],
    "positive_rate": 0.1130,
    "n_experiments": len(df),
    "top_features":  top_feats,
    "wilcoxon":      wilcoxon,
}

meta_path = os.path.join(OUT_DIR, "metadata_r01.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"\nSaved metadata -> {meta_path}")

# Now generate the abstract
script = os.path.join(R01_DIR, "generate_abstract_r01.py")
os.system(f"python \"{script}\"")
