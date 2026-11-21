"""Decision Tree + Random Forest experiments on the shared split.

Every run is logged to results/results.csv and the best tree/forest validation predictions are
saved to outputs/preds/.

Usage (from repo root):
    python experiments/tree_experiments.py                # trees + forest, both tasks
    python experiments/tree_experiments.py --fisher       # forest on Fisher (LDA) directions
"""
import argparse
from time import perf_counter

import _utils  # noqa: F401  (sets sys.path)

import numpy as np

from src.common import get_split, log_result, save_predictions, set_seed
from src.common.config import SEED
from src.models import DecisionTree, RandomForest

# Depth 5 -> 25 shows underfitting turning into overfitting; then one change at a time from
# depth 20: bigger min split, smaller min leaf, and entropy instead of gini.
TREE_CONFIGS = [  # (max_depth, min_samples_split, min_samples_leaf, criterion)
    (5, 10, 5, "gini"), (10, 10, 5, "gini"), (15, 10, 5, "gini"), (20, 10, 5, "gini"),
    (25, 10, 5, "gini"), (20, 20, 5, "gini"), (20, 10, 2, "gini"), (20, 10, 5, "entropy"),
]


def project_fisher(X_train, X_val, y_train, shrinkage=0.10, class_aligned=False):
    """Regularized Fisher (LDA) directions, fit on the training fold only.

    Grey area under the rules: it's supervised dimensionality reduction. Report raw results too.
    """
    train = X_train.astype(np.float64)
    labels, encoded, counts = np.unique(y_train, return_inverse=True, return_counts=True)
    means = np.array([train[encoded == i].mean(axis=0) for i in range(len(labels))])
    center = train.mean(axis=0)
    # Shrink within-class covariance toward a scaled identity for stability.
    residuals = train - means[encoded]
    covariance = residuals.T @ residuals / len(train)
    target = np.trace(covariance) / covariance.shape[0]
    regularized = (1 - shrinkage) * covariance + shrinkage * target * np.eye(len(covariance))
    if class_aligned:
        projection = np.linalg.solve(regularized, (means - center).T)
    else:
        # Whiten within-class variation, then find directions of class-mean variation.
        factor = np.linalg.cholesky(regularized)
        offsets = np.linalg.solve(factor, (means - center).T).T
        _, _, axes = np.linalg.svd(offsets * np.sqrt(counts[:, None]), full_matrices=False)
        projection = np.linalg.solve(factor.T, axes[:len(labels) - 1].T)
    return ((train - center) @ projection).astype("float32"), \
        ((X_val - center) @ projection).astype("float32")


def run_trees(task, X_tr, y_tr, X_val, y_val, author):
    best = (-1, None, None)
    for depth, split, leaf, criterion in TREE_CONFIGS:
        tree = DecisionTree(max_depth=depth, min_samples_split=split, min_samples_leaf=leaf,
                            criterion=criterion, random_state=SEED)
        t = perf_counter()
        tree.fit(X_tr, y_tr)
        fit_s = perf_counter() - t
        t = perf_counter()
        val_pred = tree.predict(X_val)
        pred_s = perf_counter() - t
        train_acc, val_acc = np.mean(tree.predict(X_tr) == y_tr), np.mean(val_pred == y_val)
        log_result("decision_tree", task, tree.get_params(), train_acc=train_acc, val_acc=val_acc,
                   train_time=fit_s, predict_time=pred_s, author=author, notes="raw features")
        if val_acc > best[0]:
            best = (val_acc, val_pred, tree.predict_proba(X_val))
    save_predictions("tree_best", task, "val", best[1], best[2])


def run_forest(task, X_tr, y_tr, X_val, y_val, author, fisher):
    # Forest trees can be deep (averaging removes most of their overfitting). Image uses fewer
    # trees (25) because a 50-class tree is much slower to grow than a 3-class one.
    depth = 20 if task == "image" else 15
    configs = [("raw", depth, 5, "sqrt", 25 if task == "image" else 50)]
    if fisher:
        configs = [("fisher", 18, 5, "sqrt", 50),
                   ("fisher-class", 18, 5, "sqrt", 50) if task == "image"
                   else ("fisher", 5, 10, None, 50)]

    best = (-1, None, None, None)
    for rep, depth, leaf, features, n_trees in configs:
        train, val = X_tr, X_val
        if rep.startswith("fisher"):
            train, val = project_fisher(X_tr, X_val, y_tr, class_aligned=(rep == "fisher-class"))
        forest = RandomForest(n_estimators=n_trees, max_depth=depth, min_samples_split=2 * leaf,
                              min_samples_leaf=leaf, max_features=features, random_state=SEED,
                              verbose=False)
        t = perf_counter()
        forest.fit(train, y_tr)
        fit_s = perf_counter() - t
        # Score the forest after 10, 25, 50, ... trees, with hard and soft voting: one fit
        # gives the whole "more trees" curve.
        votes = {v: [np.zeros((len(X), len(forest.classes_))) for X in (train, val)]
                 for v in ("hard", "soft")}
        for i, tree in enumerate(forest.trees, start=1):
            cols = np.searchsorted(forest.classes_, tree.classes_)
            for j, X in enumerate((train, val)):
                p = tree.predict_proba(X)
                votes["hard"][j][np.arange(len(X)), cols[p.argmax(axis=1)]] += 1
                votes["soft"][j][:, cols] += p
            if i not in {10, 25, 50, n_trees}:
                continue
            for voting, (s_tr, s_val) in votes.items():
                val_pred = forest.classes_[s_val.argmax(axis=1)]
                train_acc = np.mean(forest.classes_[s_tr.argmax(axis=1)] == y_tr)
                val_acc = np.mean(val_pred == y_val)
                params = {**forest.get_params(), "n_estimators": i, "voting": voting}
                log_result("random_forest", task, params, train_acc=train_acc, val_acc=val_acc,
                           train_time=fit_s * i / n_trees, author=author, notes=f"{rep} features")
                if val_acc > best[0]:
                    best = (val_acc, val_pred, s_val / i, f"{rep} {params}")
    save_predictions("forest_fisher_best" if fisher else "forest_best", task, "val", best[1], best[2])
    print(f"[forest] {task} best val {best[0]:.4f}: {best[3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("image", "sentiment", "both"), default="both")
    ap.add_argument("--author", default="")
    ap.add_argument("--skip-trees", action="store_true")
    ap.add_argument("--fisher", action="store_true", help="Forest on train-only Fisher directions.")
    args = ap.parse_args()
    set_seed()
    for task in ("image", "sentiment") if args.task == "both" else (args.task,):
        X_tr, y_tr, X_val, y_val = get_split(task)
        if not args.skip_trees:
            run_trees(task, X_tr, y_tr, X_val, y_val, args.author)
        run_forest(task, X_tr, y_tr, X_val, y_val, args.author, args.fisher)


if __name__ == "__main__":
    main()
