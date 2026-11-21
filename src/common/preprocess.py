"""Preprocessing helpers (NumPy). ALWAYS fit on training data only.

    scaler = Standardizer().fit(X_tr)
    X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)
"""
import numpy as np


class Standardizer:
    """Zero mean, unit variance per feature (z-score)."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + self.eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() first.")
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row to unit length. After this, Euclidean ranking == cosine ranking.

    Stateless (row-wise), so no fit step is needed.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(norms, eps)).astype(np.float32)


class PCA:
    """Plain NumPy PCA via SVD. Useful for 2-D plots and optional dimensionality reduction."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[: self.n_components]
        var = S**2
        self.explained_variance_ratio_ = var[: self.n_components] / var.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) @ self.components_.T).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
