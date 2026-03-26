"""
src/features/engineering.py
Feature selection: chi-square for categorical, correlation for numerical.
Returns full and reduced feature sets.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chi2_feature_selection(X_ohe: np.ndarray, y: np.ndarray, feature_names: list, k: int = 20):
    """Select top-k features via chi-square test (non-negative values required)."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_ohe)
    selector = SelectKBest(chi2, k=min(k, X_ohe.shape[1]))
    selector.fit(X_scaled, y)
    mask = selector.get_support()
    selected = [name for name, sel in zip(feature_names, mask) if sel]
    scores = dict(zip(feature_names, selector.scores_))
    logger.info(f"Chi2 selected {len(selected)} features from {len(feature_names)}")
    return selected, scores


def correlation_filter(X_le: np.ndarray, feature_names: list, threshold: float = 0.90):
    """Drop features with pairwise Pearson correlation > threshold."""
    df = pd.DataFrame(X_le, columns=feature_names)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    keep_cols = [c for c in feature_names if c not in drop_cols]
    logger.info(f"Correlation filter: dropped {len(drop_cols)} features. Kept {len(keep_cols)}")
    return keep_cols, drop_cols


def get_feature_sets(X_ohe, y, feature_names_ohe, X_le, feature_names_le):
    """
    Returns dict:
        full_ohe, reduced_ohe, full_le, reduced_le
        (numpy arrays + corresponding name lists)
    """
    # OHE reduced: chi2 selection
    selected_ohe, chi2_scores = chi2_feature_selection(X_ohe, y, feature_names_ohe, k=25)
    ohe_idx = [feature_names_ohe.index(f) for f in selected_ohe]
    X_ohe_reduced = X_ohe[:, ohe_idx]

    # LE reduced: correlation filter
    keep_le, _ = correlation_filter(X_le, feature_names_le, threshold=0.90)
    le_idx = [feature_names_le.index(f) for f in keep_le]
    X_le_reduced = X_le[:, le_idx]

    return {
        "full_ohe": (X_ohe, feature_names_ohe),
        "reduced_ohe": (X_ohe_reduced, selected_ohe),
        "full_le": (X_le, feature_names_le),
        "reduced_le": (X_le_reduced, keep_le),
        "chi2_scores": chi2_scores,
    }
