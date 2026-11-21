"""Evaluation helpers. Library metrics are explicitly allowed by the course."""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix as _cm

from .config import FIGURES_DIR, check_task


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def per_class_report(y_true, y_pred, task: str, as_dict: bool = False):
    """Precision / recall / F1 per class (with readable names for sentiment)."""
    meta = check_task(task)
    return classification_report(
        y_true, y_pred,
        labels=list(range(meta["n_classes"])),
        target_names=meta["class_names"],
        digits=3, zero_division=0, output_dict=as_dict,
    )


def confusion_matrix(y_true, y_pred, task: str) -> np.ndarray:
    n = check_task(task)["n_classes"]
    return _cm(y_true, y_pred, labels=list(range(n)))


def top_confused_pairs(cm: np.ndarray, k: int = 10):
    """Most frequent off-diagonal (true, predicted, count) entries."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    rows, cols = np.unravel_index(flat, off.shape)
    return [(int(r), int(c), int(off[r, c])) for r, c in zip(rows, cols) if off[r, c] > 0]


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    return np.diag(cm) / np.maximum(cm.sum(axis=1), 1)


def plot_confusion(cm: np.ndarray, task: str, title: str, filename: str | None = None,
                   normalize: bool = True):
    """Save a confusion-matrix heatmap into outputs/figures/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = check_task(task)
    mat = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    size = 10 if meta["n_classes"] > 10 else 5
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if meta["n_classes"] <= 10:
        ticks = range(meta["n_classes"])
        ax.set_xticks(ticks, meta["class_names"])
        ax.set_yticks(ticks, meta["class_names"])
        for i in ticks:
            for j in ticks:
                ax.text(j, i, f"{mat[i, j]:.2f}" if normalize else int(mat[i, j]),
                        ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / (filename or f"cm_{task}_{title.replace(' ', '_')}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

