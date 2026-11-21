"""Verify the Kaggle CSVs are in data/ and look as expected.  Usage: python scripts/check_setup.py"""
import _path  # noqa: F401
import platform

import numpy as np

from src.common.config import DATA_DIR, SAMPLE_SUBMISSION_FILE, TASKS
from src.common.data import load_sample_submission, load_test, load_train


def main():
    print(f"Python {platform.python_version()} | NumPy {np.__version__} | data dir: {DATA_DIR}")
    try:
        import torch
        print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not installed (needed for the neural net): pip install torch")

    ok = True
    for task, meta in TASKS.items():
        try:
            ids, X, y = load_train(task)
            tids, Xt = load_test(task)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {task}: {e}")
            ok = False
            continue
        counts = np.bincount(y, minlength=meta["n_classes"])
        print(f"\n[{task}] train X={X.shape} test X={Xt.shape}")
        print(f"  labels {y.min()}-{y.max()}, per-class count min={counts.min()} max={counts.max()}")
        print(f"  feature mean={X.mean():.3f} std={X.std():.3f}  NaNs={int(np.isnan(X).sum())}")
        print(f"  mean row L2 norm={np.linalg.norm(X, axis=1).mean():.3f}")
        if X.shape != (meta["n_train"], meta["n_features"]):
            print(f"  WARNING: expected train shape {(meta['n_train'], meta['n_features'])}")
        if Xt.shape[0] != meta["n_test"]:
            print(f"  WARNING: expected {meta['n_test']} test rows")
        if not all(str(i).startswith(meta["test_id_prefix"]) for i in tids[:5]):
            print(f"  WARNING: test ids don't start with {meta['test_id_prefix']!r}")

    try:
        sample = load_sample_submission()
        print(f"\n[{SAMPLE_SUBMISSION_FILE}] {len(sample)} rows, columns {list(sample.columns)}")
    except FileNotFoundError:
        print(f"ERROR: {SAMPLE_SUBMISSION_FILE} missing")
        ok = False

    print("\nAll good OK" if ok else "\nFix the problems above (download CSVs from the Kaggle Data tab).")


if __name__ == "__main__":
    main()
