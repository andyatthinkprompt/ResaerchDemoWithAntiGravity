"""
src/data/download.py
Downloads the UCI Bank Marketing dataset.
Falls back to direct UCI ML repository download (no Kaggle credentials needed).
"""
import os
import urllib.request
import zipfile
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
FALLBACK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"


def download_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)
    target_csv = os.path.normpath(os.path.join(RAW_DIR, "bank.csv"))

    if os.path.exists(target_csv):
        logger.info(f"Dataset already exists at {target_csv}, skipping download.")
        return target_csv

    # Try bank-additional-full (4521 × 21) — preferred
    zip_path = os.path.join(RAW_DIR, "bank.zip")
    logger.info("Downloading UCI Bank Marketing dataset (bank-additional)...")
    try:
        urllib.request.urlretrieve(UCI_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        # bank-additional.zip → bank-additional/bank-additional-full.csv
        full_csv = os.path.join(RAW_DIR, "bank-additional", "bank-additional-full.csv")
        if os.path.exists(full_csv):
            shutil.copy(full_csv, target_csv)
            logger.info(f"Saved dataset to {target_csv}")
            return target_csv
    except Exception as e:
        logger.warning(f"Primary download failed: {e}. Trying fallback...")

    # Fallback: bank.zip (older 4521-row version)
    try:
        urllib.request.urlretrieve(FALLBACK_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        fallback_csv = os.path.join(RAW_DIR, "bank-full.csv")
        if os.path.exists(fallback_csv):
            shutil.copy(fallback_csv, target_csv)
            logger.info(f"Saved fallback dataset to {target_csv}")
            return target_csv
    except Exception as e:
        logger.error(f"Fallback download also failed: {e}")
        raise RuntimeError(
            "Could not download dataset. Please manually place bank.csv in data/raw/"
        )


if __name__ == "__main__":
    path = download_dataset()
    import pandas as pd
    df = pd.read_csv(path, sep=";")
    print(f"Dataset shape: {df.shape}")
    print(df.head(2))
