"""k-Means clustering, from scratch (NumPy). Not part of the Kaggle score.

Lloyd's algorithm: assign each point to its nearest centre, move each centre to the mean of its
points, repeat until nothing changes. Starts from k-means++ centres and keeps the best of
n_init restarts (lowest inertia), since each run only finds a local minimum.
"""
import numpy as np


def _sq_dists(X: np.ndarray, C: np.ndarray, X_sq: np.ndarray | None = None) -> np.ndarray:
    """(n, k) squared Euclidean distances via ||x||² + ||c||² - 2x·c."""
    if X_sq is None:
        X_sq = np.einsum("ij,ij->i", X, X)
    C_sq = np.einsum("ij,ij->i", C, C)
    d2 = X_sq[:, None] + C_sq[None, :] - 2.0 * (X @ C.T)
    return np.maximum(d2, 0.0)


class KMeans:
    def __init__(self, n_clusters: int, n_init: int = 5, max_iter: int = 300,
                 tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def get_params(self) -> dict:
        return {"n_clusters": self.n_clusters, "n_init": self.n_init,
                "max_iter": self.max_iter, "random_state": self.random_state}

    # ------------------------------------------------------------------ init
    def _kmeans_plus_plus(self, X, X_sq, rng):
        n = len(X)
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)
        centers[0] = X[rng.integers(n)]
        closest = _sq_dists(X, centers[:1], X_sq)[:, 0]       # D(x)² to nearest chosen centre
        for c in range(1, self.n_clusters):
            probs = closest.astype(np.float64)
            total = probs.sum()
            if total <= 0:                                    # all points identical to centres
                centers[c] = X[rng.integers(n)]
            else:
                centers[c] = X[rng.choice(n, p=probs / total)]
            closest = np.minimum(closest, _sq_dists(X, centers[c:c + 1], X_sq)[:, 0])
        return centers

    # ------------------------------------------------------------------ one run
    def _single_run(self, X, X_sq, rng):
        centers = self._kmeans_plus_plus(X, X_sq, rng)
        labels = None
        for it in range(1, self.max_iter + 1):
            d2 = _sq_dists(X, centers, X_sq)
            new_labels = np.argmin(d2, axis=1)                # assignment step
            if labels is not None and np.array_equal(new_labels, labels):
                break                                         # converged: nothing moved
            labels = new_labels

            # Update step: each centre = mean of its points. Done for all clusters at once with
            # a one-hot assignment matrix M (n × k): M.T @ X gives per-cluster sums.
            M = np.zeros((len(X), self.n_clusters), dtype=X.dtype)
            M[np.arange(len(X)), labels] = 1.0
            counts = M.sum(axis=0)
            new_centers = (M.T @ X) / np.maximum(counts, 1.0)[:, None]
            empty = np.flatnonzero(counts == 0)
            if len(empty):                                    # empty cluster → re-seed
                point_cost = d2[np.arange(len(X)), labels].copy()
                for c in empty:
                    far = int(np.argmax(point_cost))
                    new_centers[c] = X[far]
                    point_cost[far] = 0.0
            shift = np.sqrt(((new_centers - centers) ** 2).sum(axis=1)).max()
            centers = new_centers
            if shift < self.tol:
                break
        d2 = _sq_dists(X, centers, X_sq)
        labels = np.argmin(d2, axis=1)
        inertia = float(d2[np.arange(len(X)), labels].sum())
        return centers, labels, inertia, it

    # ------------------------------------------------------------------ public API
    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        X_sq = np.einsum("ij,ij->i", X, X)
        rng = np.random.default_rng(self.random_state)
        best = None
        self.all_inertias_ = []
        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._single_run(X, X_sq, rng)
            self.all_inertias_.append(inertia)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia, n_iter)
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.argmin(_sq_dists(X, self.cluster_centers_), axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_


# ---------------------------------------------------------------------- evaluation helpers
def purity(y_true: np.ndarray, clusters: np.ndarray) -> float:
    """Fraction of points whose cluster's majority label equals their own label.
    Caveat: purity rises trivially with more clusters (k = n gives purity 1), so pair it with
    ARI/NMI."""
    total = 0
    for c in np.unique(clusters):
        total += np.bincount(y_true[clusters == c]).max()
    return total / len(y_true)


def contingency_table(y_true: np.ndarray, clusters: np.ndarray, n_classes: int,
                      n_clusters: int) -> np.ndarray:
    """rows = true class, cols = cluster, entries = counts."""
    table = np.zeros((n_classes, n_clusters), dtype=int)
    np.add.at(table, (y_true, clusters), 1)
    return table


def class_cooccurrence(table: np.ndarray) -> np.ndarray:
    """How strongly two classes share clusters (class × class, in [0, 1]).

    For classes i, j: sum over clusters of min(frac_i, frac_j), where frac_i is the fraction of
    class i's points in that cluster. 1 = identical cluster distributions, 0 = never share.
    """
    frac = table / np.maximum(table.sum(axis=1, keepdims=True), 1)
    n = len(frac)
    co = np.zeros((n, n))
    for i in range(n):
        co[i] = np.minimum(frac[i][None, :], frac).sum(axis=1)
    np.fill_diagonal(co, 0.0)
    return co
