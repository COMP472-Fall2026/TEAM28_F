"""Experiments aimed at the Kaggle score, on the shared validation split.

Every run is logged to results/results.csv (model names start with "boost_") and validation
probabilities are saved to outputs/preds/ so models can be blended. Selection is on validation
only, never on the leaderboard.

Usage (from repo root):
    python experiments/score_boost.py --task image
    python experiments/score_boost.py --task sentiment
    python experiments/score_boost.py --blend      # image blend-weight table from saved predictions
"""
import argparse

import _utils  # noqa: F401  (sets sys.path)
from _utils import ROOT

import numpy as np

import mlp_experiments as M  # MLP, fit, predict (training loop with early stopping)
from src.common import (Standardizer, get_split, l2_normalize as l2, load_predictions, log_result,
                        save_predictions, set_seed)
from src.models.knn import KNN, local_mean_scores, softmax

AUTHOR = "Ashwin"
HIST = ROOT / "outputs" / "boost_histories"


def hellinger(A):
    """Square root then unit length: suits the sparse image histograms."""
    return l2(np.sqrt(A))


def record(name, task, params, y_val, proba, train_acc=None, notes=""):
    acc = float(np.mean(proba.argmax(1) == y_val))
    log_result(f"boost_{name}", task, params, val_acc=acc, train_acc=train_acc, author=AUTHOR,
               notes=notes)
    save_predictions(f"boost_{name}", task, "val", proba.argmax(1), proba)
    return acc


def blend_table():
    """Image blend weights tried on validation (reads saved predictions; trains nothing)."""
    _, _, _, y_val = get_split("image")
    p_knn = load_predictions("boost_lmknn_m2", "image", "val")[1]
    p_mlp = load_predictions("boost_mlp1024", "image", "val")[1]
    print("weight on local-mean k-NN | image val accuracy")
    for w in (1.0, 0.9, 0.8, 0.7):
        print(f"{w:25.1f} | {np.mean((w * p_knn + (1 - w) * p_mlp).argmax(1) == y_val):.4f}")


def train_torch(name, task, A, y, B, yv, config, seeds):
    probas, train_accs = [], []
    for seed in seeds:
        model, epoch, _ = M.fit(A, y, config, seed, HIST / f"{name}_{task}_{seed}.csv",
                                validation=(B, yv))
        probas.append(M.predict(model, B))
        train_accs.append(M.evaluate(model, A, y)[1])
        print(f"[{name}] seed {seed}: best epoch {epoch}, val {np.mean(probas[-1].argmax(1) == yv):.4f}",
              flush=True)
    return np.mean(probas, axis=0), float(np.mean(train_accs))


def run(task, seeds, only=None):
    X, y, Xv, yv = get_split(task)
    n_classes = int(y.max()) + 1
    results = {}
    if task == "image":
        A, B = hellinger(X), hellinger(Xv)
        notes = "hellinger features"
    else:
        sc = Standardizer().fit(X)
        A, B = sc.transform(X).astype(np.float32), sc.transform(Xv).astype(np.float32)
        notes = "standardized features"

    # 1. plain k-NN (cosine) and local-mean k-NN (skipped when --only picks torch configs)
    if not only:
        S = l2(B) @ l2(A).T  # cosine similarities, val x train
        kmax = 10 if task == "image" else 50
        knn = KNN(k=kmax, metric="cosine").fit(A, y)
        idx, d = knn.kneighbors(B, kmax)
        for k in (1, 3) if task == "image" else (50,):
            p = knn.proba_from_neighbors(idx, d, k, weighted=True)
            results[f"knn_k{k}"] = record(f"knn_k{k}", task, {"k": k, "weighted": True}, yv, p,
                                          notes=notes)
        # small m for images (1-NN works best there), large m for sentiment (k=50-100 works best)
        for m in (2, 3, 5) if task == "image" else (25, 50):
            # temp=0.02 turns the small similarity gaps into usable probabilities for blending;
            # it doesn't change which class wins
            p = softmax(local_mean_scores(S, y, n_classes, m), temp=0.02)
            results[f"lmknn_m{m}"] = record(f"lmknn_m{m}", task, {"m": m}, yv, p, notes=notes)

    # 2. linear softmax classifier and regularized MLP (PyTorch)
    sc2 = Standardizer().fit(A)
    A2, B2 = sc2.transform(A).astype(np.float32), sc2.transform(B).astype(np.float32)
    configs = {
        # linear = no hidden layer: shows how much the hidden layer actually adds
        "linear": dict(name="linear", hidden_sizes=[], dropout=0.0, input_dropout=0.2,
                       weight_decay=1e-3, learning_rate=1e-3, num_classes=n_classes),
        "mlp1024": dict(name="mlp1024", hidden_sizes=[1024], dropout=0.5, input_dropout=0.2,
                        weight_decay=1e-3, learning_rate=1e-3, num_classes=n_classes),
    }
    if task == "sentiment":
        configs["mlp_teammate"] = dict(M.SENTIMENT_CONFIG)
        configs["mlp256"] = dict(name="mlp256", hidden_sizes=[256], dropout=0.5, input_dropout=0.1,
                                 weight_decay=1e-2, learning_rate=3e-4, num_classes=3)
        # Round 2: small variations of the best sentiment config (768-64-3).
        configs["mlp64_in0.1"] = dict(M.SENTIMENT_CONFIG, name="mlp64_in0.1", input_dropout=0.1)
        configs["mlp128"] = dict(M.SENTIMENT_CONFIG, name="mlp128", hidden_sizes=[128], dropout=0.5)
        configs["mlp64_7seeds"] = dict(M.SENTIMENT_CONFIG)  # same config, run with more seeds
    if only:
        configs = {k: v for k, v in configs.items() if k in only}
    for name, cfg in configs.items():
        p, tr = train_torch(name, task, A2, y, B2, yv, cfg, seeds)
        results[name] = record(name, task, {**cfg, "seeds": seeds}, yv, p, train_acc=tr,
                               notes=f"{notes}, then standardized")

    print(f"\n=== {task} validation accuracy ===")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"{k:16s} {v:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("image", "sentiment"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    ap.add_argument("--only", nargs="+", help="Run only these torch configs (skips k-NN).")
    ap.add_argument("--blend", action="store_true", help="Print the image blend-weight table.")
    args = ap.parse_args()
    if args.blend:
        return blend_table()
    if not args.task:
        ap.error("--task is required unless --blend is given")
    set_seed()
    HIST.mkdir(parents=True, exist_ok=True)
    run(args.task, args.seeds, args.only)


if __name__ == "__main__":
    main()
