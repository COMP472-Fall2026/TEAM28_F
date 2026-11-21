"""k-NN experiments for both tasks.

What it does, per task:
  1. Sweeps preprocessing {raw, standardized} × distance {euclidean, cosine}
     × k {1, 3, 5, 10, 20, 50, 100, 200} × voting {unweighted, distance-weighted}.
     Neighbours are computed ONCE per (preprocessing, distance) for the largest k, and every
     smaller k / voting scheme reuses them (see KNN.predict_from_neighbors).
  2. Logs every combination to results/results.csv (train + val accuracy, timings).
  3. Plots: val accuracy vs k (all configs) and train vs val vs k for the best config.
  4. For the best config: per-class report, confusion matrix, most-confused class pairs,
     confidence analysis, concrete error examples with their neighbours, saved val predictions.
  5. Retrains the best config on train+val and saves TEST predictions (outputs/preds/knn_best__*).
  6. Writes everything into outputs/knn_summary.md, the raw material for the report and slides.

Usage (from repo root):
    python experiments/knn_experiments.py --author Ashwin
    python experiments/knn_experiments.py --tasks sentiment --ks 1 5 15 25   # quick variant

Train accuracy is measured on a random 2,000-row subsample of the training split (to save
time). Each point counts itself as a neighbour, which is why k=1 train accuracy is 100%.
"""
import argparse
import json

import _utils  # noqa: F401  (sets sys.path + matplotlib backend)
from _utils import OUTPUTS, md_table, plt

import numpy as np

from src.common import (Standardizer, Timer, accuracy, confusion_matrix, get_full_train,
                        get_split, load_test, log_result, per_class_report, save_predictions,
                        set_seed, top_confused_pairs)
from src.common.config import FIGURES_DIR, SEED, TASKS
from src.common.evaluate import per_class_accuracy, plot_confusion
from src.common.split import get_val_ids
from src.models.knn import KNN

# raw vs standardized: does putting every feature on the same scale help distances?
# euclidean vs cosine: does vector length matter, or only direction?
PREPS = ["raw", "standardized"]
METRICS = ["euclidean", "cosine"]


def preprocess(name, X_fit, *others):
    """Fit preprocessing on X_fit (training data) and apply it to X_fit and the others."""
    if name == "raw":
        return (X_fit, *others)
    scaler = Standardizer().fit(X_fit)
    return (scaler.transform(X_fit), *[scaler.transform(o) for o in others])


def run_task(task, ks, author, train_subsample, make_test):
    meta = TASKS[task]
    names = meta["class_names"]
    X_tr, y_tr, X_val, y_val = get_split(task)
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(X_tr), size=min(train_subsample, len(X_tr)), replace=False)
    kmax = max(ks)

    rows, cache = [], {}
    for prep in PREPS:
        Xt, Xv, Xs = preprocess(prep, X_tr, X_val, X_tr[sub])
        for metric in METRICS:
            model = KNN(k=kmax, metric=metric)
            with Timer() as t_fit:
                model.fit(Xt, y_tr)
            with Timer() as t_pred:
                idx_v, d_v = model.kneighbors(Xv, kmax)
            idx_s, d_s = model.kneighbors(Xs, kmax)
            cache[(prep, metric)] = (model, idx_v, d_v)
            for k in ks:
                for weighted in (False, True):
                    val_pred = model.predict_from_neighbors(idx_v, d_v, k, weighted)
                    tr_pred = model.predict_from_neighbors(idx_s, d_s, k, weighted)
                    val_acc, tr_acc = accuracy(y_val, val_pred), accuracy(y_tr[sub], tr_pred)
                    params = {"k": k, "metric": metric, "weighted": weighted, "preprocessing": prep}
                    log_result("knn", task, params, train_acc=tr_acc, val_acc=val_acc,
                               train_time=t_fit.seconds, predict_time=t_pred.seconds,
                               author=author,
                               notes=f"neighbour search shared across k (k_max={kmax}); "
                                     f"train_acc on {len(sub)}-row subsample")
                    rows.append({**params, "train_acc": tr_acc, "val_acc": val_acc,
                                 "search_time_s": round(t_pred.seconds, 2)})

    # ---------------------------------------------------------------- plots
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for prep in PREPS:
        for metric in METRICS:
            for weighted, ls in ((False, "-"), (True, "--")):
                pts = [r for r in rows if r["preprocessing"] == prep and r["metric"] == metric
                       and r["weighted"] == weighted]
                ax.plot([r["k"] for r in pts], [r["val_acc"] for r in pts], ls, marker="o",
                        ms=3, label=f"{prep}/{metric}{' (weighted)' if weighted else ''}")
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("k (log scale)")
    ax.set_ylabel("validation accuracy")
    ax.set_title(f"k-NN on {task}: validation accuracy vs k")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"knn_{task}_k_sweep.png", dpi=150)
    plt.close(fig)

    best = max(rows, key=lambda r: (r["val_acc"], -r["k"]))
    # Always UNWEIGHTED here: with weighted voting a training point finds itself at distance 0,
    # gets an enormous weight, and train accuracy is trivially 100% for every k.
    pts = [r for r in rows if r["preprocessing"] == best["preprocessing"]
           and r["metric"] == best["metric"] and not r["weighted"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["k"] for r in pts], [r["train_acc"] for r in pts], marker="o", label="train (subsample)")
    ax.plot([r["k"] for r in pts], [r["val_acc"] for r in pts], marker="o", label="validation")
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("k (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"k-NN {task}: train vs val ({best['preprocessing']}/{best['metric']}, unweighted)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"knn_{task}_train_vs_val.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- best-model analysis
    model, idx_v, d_v = cache[(best["preprocessing"], best["metric"])]
    k, w = best["k"], best["weighted"]
    val_pred = model.predict_from_neighbors(idx_v, d_v, k, w)
    proba = model.proba_from_neighbors(idx_v, d_v, k, w)
    save_predictions("knn_best", task, "val", val_pred, proba)

    cm = confusion_matrix(y_val, val_pred, task)
    plot_confusion(cm, task, f"k-NN best ({task})", filename=f"knn_{task}_confusion.png")
    pca = per_class_accuracy(cm)
    confused = top_confused_pairs(cm, 10)

    # Confidence: fraction of the k votes going to the predicted class.
    conf = proba.max(axis=1)
    correct = val_pred == y_val
    conf_rows = []
    for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.any():
            conf_rows.append({"vote share of predicted class": f"[{lo:.1f}, {min(hi, 1):.1f}]",
                              "fraction of val samples": float(m.mean()),
                              "accuracy in this bucket": float(correct[m].mean())})

    # Concrete error examples: what did the neighbours look like?
    val_ids = get_val_ids(task)
    wrong = np.flatnonzero(~correct)
    examples = []
    for i in wrong[:8]:
        neigh = model.y_[idx_v[i, :k]]
        examples.append({"val id": str(val_ids[i]), "true": names[y_val[i]],
                         "pred": names[val_pred[i]], "vote share": float(conf[i]),
                         f"labels of the {min(k, 10)} nearest": " ".join(names[j] for j in neigh[:10]),
                         "nearest dist": float(d_v[i, 0])})

    # Effect of preprocessing / metric: best val acc for each config (any k, any weighting)
    config_rows = []
    for prep in PREPS:
        for metric in METRICS:
            sub_rows = [r for r in rows if r["preprocessing"] == prep and r["metric"] == metric]
            b = max(sub_rows, key=lambda r: r["val_acc"])
            k1 = next(r for r in sub_rows if r["k"] == min(ks) and not r["weighted"])
            config_rows.append({"preprocessing": prep, "metric": metric,
                                "best val acc": b["val_acc"], "at k": b["k"],
                                "weighted": b["weighted"], f"val acc at k={min(ks)}": k1["val_acc"],
                                "search time (s)": b["search_time_s"]})
    weight_rows = []
    for kk in ks:
        a = next(r for r in rows if r["k"] == kk and not r["weighted"]
                 and r["preprocessing"] == best["preprocessing"] and r["metric"] == best["metric"])
        b = next(r for r in rows if r["k"] == kk and r["weighted"]
                 and r["preprocessing"] == best["preprocessing"] and r["metric"] == best["metric"])
        weight_rows.append({"k": kk, "train acc (unweighted)": a["train_acc"],
                            "val acc (unweighted)": a["val_acc"],
                            "val acc (weighted)": b["val_acc"],
                            "weighted − unweighted": b["val_acc"] - a["val_acc"]})

    # ---------------------------------------------------------------- final test predictions
    if make_test:
        X_full, y_full = get_full_train(task)
        _, X_test = load_test(task)
        Xf, Xte = preprocess(best["preprocessing"], X_full, X_test)
        final = KNN(k=k, metric=best["metric"], weighted=w).fit(Xf, y_full)
        save_predictions("knn_best", task, "test", final.predict(Xte))

    # ---------------------------------------------------------------- markdown section
    md = [f"## {task.capitalize()} ({meta['n_classes']} classes, {meta['n_features']}-dim)\n",
          f"**Best config:** preprocessing=`{best['preprocessing']}`, metric=`{best['metric']}`, "
          f"k=`{k}`, weighted=`{w}` → **val acc {best['val_acc']:.4f}** "
          f"(train-subsample acc {best['train_acc']:.4f}).\n",
          f"Chance level = {1 / meta['n_classes']:.3f}.\n",
          "### Preprocessing × distance (best over k and voting)\n", md_table(config_rows),
          "\n### k sweep for the best preprocessing/distance (bias–variance)\n", md_table(weight_rows),
          f"\nFigures: `outputs/figures/knn_{task}_k_sweep.png`, `knn_{task}_train_vs_val.png`, "
          f"`knn_{task}_confusion.png`\n",
          "### Confidence vs correctness (vote share of the winning class)\n", md_table(conf_rows)]
    if task == "sentiment":
        md += ["\n### Per-class metrics\n", "```\n" + per_class_report(y_val, val_pred, task) + "```"]
    else:
        order = np.argsort(pca)
        md += ["\n### Hardest and easiest classes (per-class val accuracy)\n",
               md_table([{"class": int(c), "accuracy": float(pca[c])} for c in order[:8]]),
               "\n", md_table([{"class": int(c), "accuracy": float(pca[c])} for c in order[-5:][::-1]])]
    md += ["\n### Most confused pairs (true → predicted, count)\n",
           md_table([{"true": names[a], "predicted": names[b], "count": c} for a, b, c in confused]),
           "\n### Example errors and their neighbours\n", md_table(examples)]
    print(f"[knn] {task}: best {json.dumps({x: best[x] for x in ('preprocessing','metric','k','weighted')})}"
          f" val={best['val_acc']:.4f}")
    return "\n".join(md), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    # roughly doubling k covers "memorize" (k=1) to "very smooth" (k=200) cheaply; 100 and 200
    # were added because sentiment was still improving at k=50
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 20, 50, 100, 200])
    ap.add_argument("--author", default="")
    ap.add_argument("--train-subsample", type=int, default=2000)
    ap.add_argument("--no-test", action="store_true", help="skip retrain + test predictions")
    args = ap.parse_args()
    set_seed()

    sections, bests = [], {}
    for task in args.tasks:
        md, best = run_task(task, sorted(args.ks), args.author, args.train_subsample, not args.no_test)
        sections.append(md)
        bests[task] = best

    summary = ["# k-NN results (auto-generated by experiments/knn_experiments.py)\n",
               "| Task | Best config | Val acc |", "|---|---|---|"]
    for t, b in bests.items():
        summary.append(f"| {t} | {b['preprocessing']}/{b['metric']}, k={b['k']}, "
                       f"{'weighted' if b['weighted'] else 'unweighted'} | {b['val_acc']:.4f} |")
    out = OUTPUTS / "knn_summary.md"
    out.write_text("\n".join(summary) + "\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"[knn] summary -> {out}")


if __name__ == "__main__":
    main()
