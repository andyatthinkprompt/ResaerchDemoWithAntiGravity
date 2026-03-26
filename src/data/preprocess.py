"""
src/data/preprocess.py
Loads raw bank.csv and produces encoded feature matrices.
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
PROC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))

LEAKAGE_FEATURES = ["duration"]

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome"
]

NUMERICAL_COLS = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"
]


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load raw CSV with semicolon separator."""
    df = pd.read_csv(csv_path, sep=";")
    logger.info(f"Loaded raw data: {df.shape}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: replace 'unknown' with NaN, drop duplicates."""
    df = df.copy()
    df.replace("unknown", np.nan, inplace=True)
    initial = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {initial - len(df)} duplicate rows. Shape: {df.shape}")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert y: yes→1, no→0."""
    df = df.copy()
    df["y"] = (df["y"] == "yes").astype(int)
    logger.info(f"Target distribution:\n{df['y'].value_counts()}")
    return df


def get_feature_cols(df: pd.DataFrame, drop_leakage: bool = True):
    """Return cat/num column lists present in df."""
    all_cols = [c for c in df.columns if c != "y"]
    if drop_leakage:
        all_cols = [c for c in all_cols if c not in LEAKAGE_FEATURES]
    cats = [c for c in CATEGORICAL_COLS if c in all_cols]
    nums = [c for c in NUMERICAL_COLS if c in all_cols]
    return cats, nums


def one_hot_encode(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Apply one-hot encoding to categorical columns."""
    return pd.get_dummies(df, columns=cat_cols, drop_first=False)


def label_encode(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Apply label encoding to categorical columns."""
    df = df.copy()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    return df


def preprocess(raw_csv: str, drop_leakage: bool = True):
    """
    Returns a dict with:
        X_ohe, X_le, y, feature_names_ohe, feature_names_le
    and saves processed files.
    """
    os.makedirs(PROC_DIR, exist_ok=True)
    df = load_raw(raw_csv)
    df = clean(df)
    df = encode_target(df)

    cat_cols, num_cols = get_feature_cols(df, drop_leakage=drop_leakage)
    feature_cols = cat_cols + num_cols

    # Fill NaN: mode for categoricals, median for numericals
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    y = df["y"].values

    # One-hot encoded version
    df_ohe = one_hot_encode(df[feature_cols + ["y"]], cat_cols)
    X_ohe = df_ohe.drop("y", axis=1)
    feature_names_ohe = list(X_ohe.columns)

    # Label encoded version
    df_le = label_encode(df[feature_cols + ["y"]], cat_cols)
    X_le = df_le.drop("y", axis=1)
    feature_names_le = list(X_le.columns)

    # Save
    X_ohe.to_csv(os.path.join(PROC_DIR, "X_ohe.csv"), index=False)
    X_le.to_csv(os.path.join(PROC_DIR, "X_le.csv"), index=False)
    pd.Series(y).to_csv(os.path.join(PROC_DIR, "y.csv"), index=False, header=["y"])

    logger.info(f"OHE features: {X_ohe.shape[1]}, LE features: {X_le.shape[1]}")

    return {
        "X_ohe": X_ohe.values.astype(float),
        "X_le": X_le.values.astype(float),
        "y": y,
        "feature_names_ohe": feature_names_ohe,
        "feature_names_le": feature_names_le,
        "df_ohe": X_ohe,
        "df_le": X_le,
    }


if __name__ == "__main__":
    result = preprocess(os.path.join(RAW_DIR, "bank.csv"))
    print("OHE X shape:", result["X_ohe"].shape)
    print("LE X shape:", result["X_le"].shape)
    print("y shape:", result["y"].shape)
