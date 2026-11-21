"""Experiment logging and prediction saving.

    with Timer() as t_fit:
        model.fit(X_tr, y_tr)
    with Timer() as t_pred:
        val_pred = model.predict(X_val)
    log_result("knn", "image", model.get_params(),
               train_acc=..., val_acc=accuracy(y_val, val_pred),
               train_time=t_fit.seconds, predict_time=t_pred.seconds,
               author="Ashwin", notes="L2-normalized, cosine")
    save_predictions("knn_k5_cos", "image", "val", val_pred)
"""
import csv
import json
import time
from datetime import datetime

import numpy as np

from .config import PREDS_DIR, RESULTS_CSV

FIELDS = ["timestamp", "author", "model", "task", "params", "train_acc", "val_acc",
          "train_time_s", "predict_time_s", "notes"]


class Timer:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False


def log_result(model: str, task: str, params: dict, *, val_acc: float,
               train_acc: float | None = None, train_time: float | None = None,
               predict_time: float | None = None, author: str = "", notes: str = "") -> None:
    """Append one experiment row to results/results.csv."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "author": author,
        "model": model,
        "task": task,
        "params": json.dumps(params, sort_keys=True, default=str),
        "train_acc": "" if train_acc is None else round(float(train_acc), 5),
        "val_acc": round(float(val_acc), 5),
        "train_time_s": "" if train_time is None else round(train_time, 3),
        "predict_time_s": "" if predict_time is None else round(predict_time, 3),
        "notes": notes,
    }
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)
    ta = f"train={row['train_acc']} " if train_acc is not None else ""
    print(f"[log] {model:<22} {task:<9} {ta}val={row['val_acc']}  {row['params']}")


def save_predictions(name: str, task: str, split: str, preds, proba=None) -> None:
    """Save predictions to outputs/preds/{name}__{task}__{split}.npz.

    split: "val" (for error analysis / model comparison) or "test" (for submission).
    """
    assert split in {"val", "test"}
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDS_DIR / f"{name}__{task}__{split}.npz"
    data = {"preds": np.asarray(preds, dtype=np.int64)}
    if proba is not None:
        data["proba"] = np.asarray(proba, dtype=np.float32)
    np.savez(path, **data)


def load_predictions(name: str, task: str, split: str):
    d = np.load(PREDS_DIR / f"{name}__{task}__{split}.npz")
    return d["preds"], (d["proba"] if "proba" in d else None)
