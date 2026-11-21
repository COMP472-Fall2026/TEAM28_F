"""Loading the Kaggle CSVs into NumPy arrays.

    ids, X, y = load_train("image")      # X: (20000, 500) float32, y: (20000,) int64
    ids, X    = load_test("sentiment")   # X: (5000, 768) float32
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import DATA_DIR, SAMPLE_SUBMISSION_FILE, check_task


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("feature_")]
    # Sort numerically so feature_10 comes after feature_9, not after feature_1.
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


@lru_cache(maxsize=None)
def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_train(task: str):
    """Return (ids, X, y) for the task's training file."""
    meta = check_task(task)
    path = DATA_DIR / meta["train_file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download it from Kaggle into {DATA_DIR}/")
    df = _read(str(path))
    feats = _feature_columns(df)
    if len(feats) != meta["n_features"]:
        raise ValueError(f"{path.name}: expected {meta['n_features']} features, found {len(feats)}")
    X = df[feats].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    return df["id"].to_numpy(), X, y


def load_test(task: str):
    """Return (ids, X) for the task's test file. Keep ids. Submissions are matched by id."""
    meta = check_task(task)
    path = DATA_DIR / meta["test_file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download it from Kaggle into {DATA_DIR}/")
    df = _read(str(path))
    feats = _feature_columns(df)
    X = df[feats].to_numpy(dtype=np.float32)
    return df["id"].to_numpy(), X


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / SAMPLE_SUBMISSION_FILE)
