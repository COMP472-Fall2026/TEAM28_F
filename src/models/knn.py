"""k-Nearest Neighbours, from scratch (NumPy).

fit() stores the training set; predict() finds the k closest training points (euclidean or
cosine distance) and lets them vote, optionally weighted by 1/distance. Distances are computed
in batches with one matrix multiply per batch, and kneighbors() is computed once for the
largest k so a whole k sweep reuses it.
"""
import numpy as np

from .base import BaseClassifier

_EPS = 1e-12


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), _EPS)


def local_mean_scores(S, y_tr, n_classes, m):
    """Local-mean k-NN: per class, average the m highest similarities to that class's training
    points, so one close point of the wrong class counts for less than in plain 1-NN.
    S is the (queries x training) cosine-similarity matrix."""
    scores = np.empty((len(S), n_classes), dtype=np.float32)
    for c in range(n_classes):
        Sc = S[:, y_tr == c]
        scores[:, c] = np.partition(Sc, -m, axis=1)[:, -m:].mean(axis=1)
    return scores


def softmax(Z, temp):
    """Turn scores into probabilities; a small temp makes them sharper (used for blending)."""
    Z = (Z - Z.max(1, keepdims=True)) / temp
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


class KNN(BaseClassifier):
    PARAMS = ("k", "metric", "weighted")

    def __init__(self, k: int = 5, metric: str = "euclidean", weighted: bool = False,
                 batch_size: int = 500):
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.metric = metric
        self.weighted = weighted
        self.batch_size = batch_size

    # ------------------------------------------------------------------ training
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Memorize the training set (plus precomputed quantities that speed up predict)."""
        X = np.asarray(X, dtype=np.float32)
        self.y_ = np.asarray(y, dtype=np.int64)
        self.n_classes_ = int(self.y_.max()) + 1
        if self.metric == "cosine":
            self.X_ = _l2_normalize(X)
        else:
            self.X_ = X
            self.sq_norms_ = np.einsum("ij,ij->i", X, X)       # ||x||² for every training row
        return self

    # ------------------------------------------------------------------ distances
    def _distances(self, Xq: np.ndarray) -> np.ndarray:
        """(n_queries, n_train) distance matrix for ONE batch of queries."""
        Xq = np.asarray(Xq, dtype=np.float32)
        if self.metric == "cosine":
            sims = _l2_normalize(Xq) @ self.X_.T                # cosine similarity
            # cosine distance in [0, 2]. Clamp because float32 rounding can give a point a
            # tiny NEGATIVE distance to itself, which would flip its 1/d vote weight negative.
            return np.maximum(1.0 - sims, 0.0)
        q_sq = np.einsum("ij,ij->i", Xq, Xq)
        d2 = q_sq[:, None] + self.sq_norms_[None, :] - 2.0 * (Xq @ self.X_.T)
        np.maximum(d2, 0.0, out=d2)        # floating-point error can make tiny negatives
        return np.sqrt(d2)                 # true distance (only matters for weighted voting)

    def kneighbors(self, X: np.ndarray, k: int | None = None):
        """Indices and distances of the k nearest training points, sorted nearest-first.

        Returns (idx, dist), both of shape (n_queries, k).
        """
        k = min(k or self.k, len(self.y_))
        all_idx, all_dist = [], []
        for start in range(0, len(X), self.batch_size):
            D = self._distances(X[start:start + self.batch_size])
            # argpartition: the k smallest end up in the first k columns (unordered)
            part = np.argpartition(D, k - 1, axis=1)[:, :k]
            part_d = np.take_along_axis(D, part, axis=1)
            order = np.argsort(part_d, axis=1)                  # sort just those k
            all_idx.append(np.take_along_axis(part, order, axis=1))
            all_dist.append(np.take_along_axis(part_d, order, axis=1))
        return np.vstack(all_idx), np.vstack(all_dist)

    # ------------------------------------------------------------------ voting
    def _vote_matrix(self, idx, dist, k, weighted, tiebreak):
        """(n_queries, n_classes) vote totals using the first k neighbours."""
        idx, dist = idx[:, :k], dist[:, :k]
        labels = self.y_[idx]                                  # (n, k) neighbour labels
        w = 1.0 / (dist + 1e-8) if weighted else np.ones_like(dist, dtype=np.float64)
        if tiebreak:
            ranks = np.arange(k)[None, :]
            w = w + 1e-6 * (k - ranks)                         # nearer neighbour → bigger bonus
        votes = np.zeros((len(idx), self.n_classes_), dtype=np.float64)
        rows = np.repeat(np.arange(len(idx)), k)
        np.add.at(votes, (rows, labels.ravel()), w.ravel())
        return votes

    def predict_from_neighbors(self, idx, dist, k: int | None = None,
                               weighted: bool | None = None) -> np.ndarray:
        """Predict from precomputed neighbours (lets a k sweep reuse one distance pass)."""
        k = k or self.k
        weighted = self.weighted if weighted is None else weighted
        return np.argmax(self._vote_matrix(idx, dist, k, weighted, tiebreak=True), axis=1)

    def proba_from_neighbors(self, idx, dist, k: int | None = None,
                             weighted: bool | None = None) -> np.ndarray:
        k = k or self.k
        weighted = self.weighted if weighted is None else weighted
        votes = self._vote_matrix(idx, dist, k, weighted, tiebreak=False)
        return votes / votes.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ public API
    def predict(self, X: np.ndarray) -> np.ndarray:
        idx, dist = self.kneighbors(X, self.k)
        return self.predict_from_neighbors(idx, dist)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Fraction of (weighted) votes per class: a rough confidence, not calibrated."""
        idx, dist = self.kneighbors(X, self.k)
        return self.proba_from_neighbors(idx, dist)
