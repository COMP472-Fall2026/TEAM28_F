"""Project-wide paths, seed, and task metadata."""
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
SPLITS_DIR = ROOT / "splits"
RESULTS_CSV = ROOT / "results" / "results.csv"
PREDS_DIR = ROOT / "outputs" / "preds"
SUBMISSIONS_DIR = ROOT / "outputs" / "submissions"
FIGURES_DIR = ROOT / "outputs" / "figures"

SEED = 42
VAL_SIZE = 0.2

TASKS = {
    "image": {
        "train_file": "image_train.csv",
        "test_file": "image_test.csv",
        "n_classes": 50,
        "n_features": 500,
        "n_train": 20_000,
        "n_test": 5_000,
        "test_id_prefix": "img_",
        "class_names": [str(i) for i in range(50)],
    },
    "sentiment": {
        "train_file": "sentiment_train.csv",
        "test_file": "sentiment_test.csv",
        "n_classes": 3,
        "n_features": 768,
        "n_train": 12_000,
        "n_test": 5_000,
        "test_id_prefix": "text_",
        "class_names": ["Negative", "Neutral", "Positive"],
    },
}
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def check_task(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {list(TASKS)}")
    return TASKS[task]
