"""The single, shared train/validation split.

Run `python scripts/make_splits.py` ONCE and commit `splits/`. Everyone then calls
`get_split(task)`, so all results in results.csv are comparable.

    X_tr, y_tr, X_val, y_val = get_split("image")
    X_full, y_full = get_full_train("image")   # train+val, for the final retrain
"""
import numpy as np
from sklearn.model_selection import train_test_split  # allowed: splitting utility

from .config import SEED, SPLITS_DIR, VAL_SIZE, check_task
from .data import load_train


def _split_path(task: str):
    return SPLITS_DIR / f"{task}_split.npz"


def make_split(task: str, val_size: float = VAL_SIZE, seed: int = SEED, overwrite: bool = False):
    """Create and save a stratified train/val split of row indices."""
    check_task(task)
    path = _split_path(task)
    if path.exists() and not overwrite:
        print(f"[split] {path} already exists. Keeping it (pass overwrite=True to redo).")
        return path
    _, _, y = load_train(task)
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_size, random_state=seed, stratify=y
    )
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_idx=np.sort(train_idx), val_idx=np.sort(val_idx),
             seed=seed, val_size=val_size)
    print(f"[split] {task}: {len(train_idx)} train / {len(val_idx)} val -> {path}")
    return path


def get_split_indices(task: str):
    path = _split_path(task)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run `python scripts/make_splits.py` (once) or pull splits/ from git."
        )
    d = np.load(path)
    return d["train_idx"], d["val_idx"]


def get_split(task: str):
    """Return X_tr, y_tr, X_val, y_val using the saved shared split."""
    _, X, y = load_train(task)
    tr, va = get_split_indices(task)
    return X[tr], y[tr], X[va], y[va]


def get_val_ids(task: str):
    """IDs of the validation rows (for looking up specific error examples)."""
    ids, _, _ = load_train(task)
    _, va = get_split_indices(task)
    return ids[va]


def get_full_train(task: str):
    """All labelled data (train + val). Use this for the final retrain only."""
    _, X, y = load_train(task)
    return X, y
