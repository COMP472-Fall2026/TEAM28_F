"""Clustering experiments with our from-scratch k-Means. Not on Kaggle.

Questions it answers (from the project spec):
  - Do sentiment embeddings naturally separate into Negative / Neutral / Positive?
  - Do visually similar objects end up in the same clusters? Which classes get grouped together?
  - How does changing the number of clusters change the results?
  - How closely do the discovered clusters correspond to the true labels?

Setup: k-Means is unsupervised. It is fit on ALL labelled training rows (train+val), and the
labels are used ONLY afterwards to score the clusters (purity, ARI, NMI, contingency tables).
Each experiment is repeated for three preprocessing choices: raw, standardized (z-score), and
L2-normalized (unit length, so k-Means effectively compares directions, like cosine).

Outputs: outputs/clustering_summary.md, outputs/clustering_results.csv, figures in outputs/figures/.

Usage (from repo root):
    python experiments/clustering_experiments.py
    python experiments/clustering_experiments.py --tasks sentiment --n-init 3
"""
import argparse
import csv

import _utils  # noqa: F401
from _utils import OUTPUTS, md_table, plt

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # metrics are allowed

from src.common import PCA, Standardizer, Timer, get_full_train, l2_normalize, set_seed
from src.common.config import FIGURES_DIR, SEED, TASKS
from src.models.kmeans import KMeans, class_cooccurrence, contingency_table, purity

# k-Means uses euclidean distance, so feature scaling changes which points look close;
# l2 (unit length) makes it compare directions, like cosine k-NN
PREPS = ["raw", "standardized", "l2"]
# k around the true number of classes (3 / 50), to see if any k matches the labels better
K_SWEEP = {"sentiment": [2, 3, 4, 5, 6, 8, 10], "image": [10, 25, 50, 75, 100]}


def preprocess(name, X):
    if name == "raw":
        return X
    if name == "standardized":
        return Standardizer().fit_transform(X)
    return l2_normalize(X)


def score(y, labels):
    return {"purity": purity(y, labels), "ARI": adjusted_rand_score(y, labels),
            "NMI": normalized_mutual_info_score(y, labels)}


def run_task(task, n_init, csv_rows, max_iter):
    meta = TASKS[task]
    names, n_classes = meta["class_names"], meta["n_classes"]
    # unsupervised, so nothing is tuned on validation: fit on all labelled rows and use the
    # labels only afterwards, to score the clusters
    X_all, y = get_full_train(task)
    md = [f"## {task.capitalize()} ({n_classes} true classes, {len(y)} points)\n"]

    # ---------------------------------------------------- 1. k = number of classes, per prep
    main_rows, fits = [], {}
    for prep in PREPS:
        X = preprocess(prep, X_all)
        with Timer() as t:
            km = KMeans(n_classes, n_init=n_init, max_iter=max_iter, random_state=SEED).fit(X)
        s = score(y, km.labels_)
        sizes = np.bincount(km.labels_, minlength=n_classes)
        fits[prep] = (X, km)
        row = {"task": task, "preprocessing": prep, "k": n_classes, **s, "inertia": km.inertia_,
               "iterations": km.n_iter_, "time_s": round(t.seconds, 2),
               "smallest cluster": int(sizes.min()), "largest cluster": int(sizes.max())}
        main_rows.append(row)
        csv_rows.append(row)
        print(f"[cluster] {task} {prep:<12} k={n_classes} purity={s['purity']:.3f} "
              f"ARI={s['ARI']:.3f} NMI={s['NMI']:.3f}")
    # NMI picks the best preprocessing: unlike purity it doesn't reward using more clusters
    best_prep = max(main_rows, key=lambda r: r["NMI"])["preprocessing"]
    X, km = fits[best_prep]
    md += [f"### k = {n_classes} (the true number of classes), by preprocessing\n",
           md_table(main_rows, ["preprocessing", "purity", "ARI", "NMI", "inertia", "iterations",
                                "smallest cluster", "largest cluster", "time_s"]),
           f"\nChance purity ≈ {np.bincount(y).max() / len(y):.3f} (largest class share). "
           f"ARI ≈ 0 and NMI ≈ 0 mean 'no better than random'. **Best by NMI: `{best_prep}`.**\n"]

    table = contingency_table(y, km.labels_, n_classes, n_classes)

    # ---------------------------------------------------- 2. task-specific structure analysis
    if task == "sentiment":
        rows = []
        for c in range(n_classes):
            col = table[:, c]
            rows.append({"cluster": c, "size": int(col.sum()),
                         **{f"% {names[i]}": float(col[i] / max(col.sum(), 1)) for i in range(n_classes)},
                         "majority": names[int(col.argmax())]})
        md += [f"### Cluster composition ({best_prep}, k=3)\n", md_table(rows, floatfmt=".3f"),
               "\nIf the clusters followed sentiment, each row would be dominated by one label. "
               "Look at whether Neutral gets its own cluster or is split between the others.\n"]
        # where does each class go?
        rows = []
        for i in range(n_classes):
            r = table[i] / table[i].sum()
            rows.append({"true class": names[i], **{f"→ cluster {c}": float(r[c]) for c in range(n_classes)}})
        md += ["#### Where each sentiment class goes\n", md_table(rows, floatfmt=".3f")]
        _pca_plot(X, y, km.labels_, names, task, best_prep)
        md += [f"\nFigure: `outputs/figures/cluster_{task}_pca.png` (true labels vs clusters in 2-D PCA).\n"]
    else:
        frac = table / table.sum(axis=1, keepdims=True)
        dominant_share = frac.max(axis=1)
        order = np.argsort(dominant_share)
        md += [f"### How 'clean' each class is ({best_prep}, k=50)\n",
               "Share of a class's points that fall in its most common cluster.\n",
               "Most coherent classes:\n",
               md_table([{"class": int(c), "share in top cluster": float(dominant_share[c]),
                          "top cluster": int(frac[c].argmax())} for c in order[::-1][:8]]),
               "\nMost scattered classes:\n",
               md_table([{"class": int(c), "share in top cluster": float(dominant_share[c]),
                          "top cluster": int(frac[c].argmax())} for c in order[:8]])]
        co = class_cooccurrence(table)
        iu = np.triu_indices(n_classes, 1)
        top = np.argsort(co[iu])[::-1][:12]
        pairs = [(int(iu[0][t]), int(iu[1][t]), float(co[iu][t])) for t in top]
        md += ["\n### Classes most often grouped together\n",
               "Overlap score in [0, 1]: how similarly two classes are spread over the clusters.\n",
               md_table([{"class A": a, "class B": b, "overlap": v} for a, b, v in pairs])]
        # Link to k-NN mistakes, if the k-NN experiment has been run
        knn_path = OUTPUTS / "preds" / "knn_best__image__val.npz"
        if knn_path.exists():
            from src.common.evaluate import confusion_matrix, top_confused_pairs
            from src.common.split import get_split
            _, _, _, y_val = get_split(task)
            knn_pred = np.load(knn_path)["preds"]
            confused = top_confused_pairs(confusion_matrix(y_val, knn_pred, task), 20)
            knn_pairs = {tuple(sorted((a, b))) for a, b, _ in confused}
            shared = [p for p in pairs if (p[0], p[1]) in knn_pairs]
            md += [f"\n**Link to k-NN errors:** {len(shared)} of the top-{len(pairs)} co-clustered "
                   f"pairs are also among k-NN's top-20 confused pairs: "
                   f"{', '.join(f'({a},{b})' for a, b, _ in shared) or 'none'}. "
                   "Classes that share clusters sit close together in feature space, which is "
                   "exactly where a distance-based classifier makes mistakes.\n"]
        _heatmap(table, task, best_prep)
        md += [f"\nFigure: `outputs/figures/cluster_{task}_contingency.png` "
               "(rows = true class, columns = cluster, sorted by each class's top cluster).\n"]

    # ---------------------------------------------------- 3. sweep over k
    sweep = []
    for k in K_SWEEP[task]:
        km_k = KMeans(k, n_init=n_init, max_iter=max_iter, random_state=SEED).fit(X)
        s = score(y, km_k.labels_)
        row = {"task": task, "preprocessing": best_prep, "k": k, **s, "inertia": km_k.inertia_,
               "iterations": km_k.n_iter_}
        sweep.append(row)
        csv_rows.append(row)
    _sweep_plot(sweep, task, best_prep, n_classes)
    md += [f"### Changing the number of clusters ({best_prep})\n",
           md_table(sweep, ["k", "purity", "ARI", "NMI", "inertia", "iterations"]),
           f"\nFigure: `outputs/figures/cluster_{task}_k_sweep.png`. Inertia always falls as k "
           "grows, so look for an 'elbow'. Purity also rises automatically with k, which is why "
           "ARI/NMI (which penalize over-splitting) are the fairer comparison.\n"]
    return "\n".join(md)


def _pca_plot(X, y, labels, names, task, prep, n=3000):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    pca = PCA(2).fit(X)
    Z = pca.transform(X[idx])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, lab, title, lname in ((axes[0], y[idx], "True labels", names),
                                  (axes[1], labels[idx], "k-Means clusters", None)):
        for c in np.unique(lab):
            m = lab == c
            ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.5,
                       label=lname[c] if lname else f"cluster {c}")
        ax.set_title(f"{title} ({prep})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cluster_{task}_pca.png", dpi=150)
    plt.close(fig)


def _heatmap(table, task, prep):
    frac = table / table.sum(axis=1, keepdims=True)
    # Sort classes by their dominant cluster, then order clusters by first appearance,
    # so classes that share a cluster end up next to each other (blocks on the diagonal).
    row_order = np.argsort(frac.argmax(axis=1), kind="stable")
    cols = list(dict.fromkeys(frac[row_order].argmax(axis=1)))
    cols += [c for c in range(table.shape[1]) if c not in cols]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(frac[row_order][:, cols], cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(row_order)), [str(r) for r in row_order], fontsize=6)
    ax.set_xticks(range(len(cols)), [str(c) for c in cols], fontsize=6, rotation=90)
    ax.set_ylabel("true class")
    ax.set_xlabel("cluster")
    ax.set_title(f"{task}: fraction of each class in each cluster ({prep})")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cluster_{task}_contingency.png", dpi=150)
    plt.close(fig)


def _sweep_plot(rows, task, prep, n_classes):
    ks = [r["k"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ks, [r["inertia"] for r in rows], "k-o", ms=3, label="inertia")
    ax1.set_xlabel("number of clusters k")
    ax1.set_ylabel("inertia (within-cluster SSE)")
    ax2 = ax1.twinx()
    for m, st in (("purity", "-s"), ("ARI", "-^"), ("NMI", "-o")):
        ax2.plot(ks, [r[m] for r in rows], st, ms=3, label=m)
    ax2.set_ylabel("agreement with true labels")
    ax1.axvline(n_classes, color="grey", ls=":", lw=1)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax1.set_title(f"k-Means on {task} ({prep}): effect of k (dotted = true #classes)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cluster_{task}_k_sweep.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    # 3 restarts: enough to avoid a bad local minimum while keeping the run to ~5 minutes
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=200)
    args = ap.parse_args()
    set_seed()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_rows, sections = [], []
    for task in args.tasks:
        sections.append(run_task(task, args.n_init, csv_rows, args.max_iter))

    with open(OUTPUTS / "clustering_results.csv", "w", newline="") as f:
        fields = sorted({k for r in csv_rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)
    out = OUTPUTS / "clustering_summary.md"
    out.write_text("# Clustering results (auto-generated by experiments/clustering_experiments.py)\n\n"
                   "k-Means (from scratch, k-means++ init, "
                   f"{args.n_init} restarts) fit on all labelled training rows. Labels used only "
                   "for scoring.\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"[cluster] summary -> {out}")


if __name__ == "__main__":
    main()
