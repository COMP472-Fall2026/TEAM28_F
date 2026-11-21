"""Build the final Kaggle submission from the models picked on validation (experiments/score_boost.py).

    image     = 0.7 * local-mean k-NN (m=2) + 0.3 * MLP 1024_reg (3 seeds), both on Hellinger features
    sentiment = MLP 768-64-3 (3 seeds), standardized features

Each model is first run on the shared train/val split to check that this exact code reproduces
the validation scores, then refit on ALL labelled rows (train + val) for test predictions. MLPs
train for the epoch count early stopping picked on validation (outputs/boost_histories/).

Usage (from repo root):  python experiments/make_final_submission.py
"""
import _utils  # noqa: F401  (sets sys.path)

import numpy as np
import pandas as pd

from score_boost import HIST, M, hellinger
from src.models.knn import local_mean_scores, softmax
from src.common import (Standardizer, build_submission, get_full_train, get_split, load_test,
                        log_result, save_predictions, set_seed)

SEEDS = [42, 123, 2026]
IMAGE_MLP = dict(name="mlp1024", hidden_sizes=[1024], dropout=0.5, input_dropout=0.2,
                 weight_decay=1e-3, learning_rate=1e-3, num_classes=50)
SENT_MLP = dict(M.SENTIMENT_CONFIG)
W_KNN = 0.7  # image blend weight, chosen on validation: python experiments/score_boost.py --blend


def best_epoch(name, task, seed):
    h = pd.read_csv(HIST / f"{name}_{task}_{seed}.csv")
    return int(h.loc[h.val_accuracy.idxmax(), "epoch"])


def mlp_proba(name, task, config, A, y, B):
    """Average of 3 seeds, each trained for its validation-chosen epoch count."""
    sc = Standardizer().fit(A)
    A, B = sc.transform(A).astype(np.float32), sc.transform(B).astype(np.float32)
    probas = []
    for seed in SEEDS:
        model, _, _ = M.fit(A, y, config, seed, HIST / f"final_{name}_{task}_{seed}.csv",
                            epochs=best_epoch(name, task, seed))
        probas.append(M.predict(model, B))
    return np.mean(probas, axis=0)


def image_proba(X_fit, y_fit, X_pred):
    A, B = hellinger(X_fit), hellinger(X_pred)
    p_knn = softmax(local_mean_scores(B @ A.T, y_fit, 50, m=2), temp=0.02)
    return W_KNN * p_knn + (1 - W_KNN) * mlp_proba("mlp1024", "image", IMAGE_MLP, A, y_fit, B)


def sentiment_proba(X_fit, y_fit, X_pred):
    return mlp_proba("mlp_teammate", "sentiment", SENT_MLP, X_fit, y_fit, X_pred)


def main():
    set_seed()
    test_preds = {}
    for task, fn, expected in (("image", image_proba, 0.626), ("sentiment", sentiment_proba, 0.683)):
        X_tr, y_tr, X_val, y_val = get_split(task)
        val_acc = float(np.mean(fn(X_tr, y_tr, X_val).argmax(1) == y_val))
        print(f"[final] {task} pipeline on train/val split: val {val_acc:.4f} (expected ~{expected})")
        assert abs(val_acc - expected) < 0.01, "pipeline does not reproduce the validation score"
        log_result(f"final_{task}", task, {"seeds": SEEDS, "w_knn": W_KNN if task == "image" else None},
                   val_acc=val_acc, author="Ashwin",
                   notes="fixed-epoch refit check on train split; test model is refit on train+val")

        X_all, y_all = get_full_train(task)
        _, X_test = load_test(task)
        p_test = fn(X_all, y_all, X_test)
        save_predictions(f"final_{task}", task, "test", p_test.argmax(1), p_test)
        test_preds[task] = p_test.argmax(1)
    build_submission(test_preds["image"], test_preds["sentiment"], name="final_blend")


if __name__ == "__main__":
    main()
