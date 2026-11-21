"""Decision Tree classifier, from scratch (NumPy).

Split search scores every threshold of a feature in one pass: sort by the feature, then
the running class counts on each side give every candidate split's impurity at once.
"""
import numpy as np

from .base import BaseClassifier


def impurity(counts, n, criterion):
    """Gini or entropy for each row of class counts; n holds the row totals."""
    p = counts / n[:, None]
    if criterion == "gini":
        return 1 - np.sum(p ** 2, axis=1)
    return -np.sum(p * np.log2(np.where(p > 0, p, 1)), axis=1)


def find_best_split(X, y, min_samples_leaf=1, max_features=None, rng=None, criterion="gini"):
    """Try every threshold of the selected features and return the split with the lowest
    weighted child impurity: (feature, threshold, impurity)."""
    if criterion not in {"gini", "entropy"}:
        raise ValueError("criterion must be 'gini' or 'entropy'.")

    n_samples, n_features = X.shape
    _, encoded = np.unique(y, return_inverse=True)        # classes at this node -> 0..c-1
    onehot = np.eye(encoded.max() + 1)[encoded]           # one row per sample, 1 in its class
    total = onehot.sum(axis=0)                            # class counts of the whole node
    n_left = np.arange(1, n_samples)                      # left-child size for each cut
    n_right = n_samples - n_left
    big_enough = (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)

    if max_features is None:
        feature_indices = np.arange(n_features)
    else:
        if rng is None:
            rng = np.random.default_rng(42)
        feature_indices = rng.choice(n_features, size=max_features, replace=False)

    best_impurity, best_feature, best_threshold = float("inf"), None, None
    for feature in feature_indices:
        order = np.argsort(X[:, feature])
        values = X[order, feature]
        # a cut can only go between two different values
        valid = big_enough & (values[:-1] < values[1:])
        if not np.any(valid):
            continue

        left_counts = np.cumsum(onehot[order], axis=0)[:-1][valid]   # class counts left of cut
        right_counts = total - left_counts
        nl, nr = n_left[valid], n_right[valid]
        impurities = (nl * impurity(left_counts, nl, criterion)
                      + nr * impurity(right_counts, nr, criterion)) / n_samples

        # earliest cut among (near-)equal scores; a new feature must be strictly better
        idx = np.flatnonzero(impurities <= impurities.min() + 1e-12)[0]
        if impurities[idx] < best_impurity - 1e-12:
            best_impurity = impurities[idx]
            best_feature = int(feature)
            split_pos = nl[idx]
            lower = float(values[split_pos - 1])
            upper = float(values[split_pos])
            midpoint = lower / 2 + upper / 2
            best_threshold = midpoint if midpoint < upper else lower

    return best_feature, best_threshold, best_impurity


class DecisionTree(BaseClassifier):
    PARAMS = ("max_depth", "min_samples_split", "min_samples_leaf", "max_features",
              "criterion", "random_state")

    def __init__(
        self,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=42,
        criterion="gini"
    ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.max_features = max_features
        self.random_state = random_state
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.criterion = criterion

        self.rng = np.random.default_rng(
            random_state
        )

        self.root = None

    # ----------------------------------------------
    # TRAIN THE TREE
    # ----------------------------------------------

    def fit(self, X, y):

        # Reset random generator for reproducibility
        self.rng = np.random.default_rng(
            self.random_state
        )

        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.root = self._grow_tree(X, encoded, depth=0)

        return self

    # ----------------------------------------------
    # RECURSIVELY GROW THE TREE
    # ----------------------------------------------

    def _grow_tree(self, X, y, depth):

        n_samples = len(y)
        counts = np.bincount(y, minlength=len(self.classes_))
        leaf = {
            "class": int(self.classes_[counts.argmax()]),
            "probabilities": counts / n_samples,
        }

        # Stopping conditions
        if (
            (
                self.max_depth is not None
                and depth >= self.max_depth
            )
            or n_samples < self.min_samples_split
            or np.count_nonzero(counts) == 1
            or n_samples < 2 * self.min_samples_leaf
        ):

            return leaf

        # Find best split
        feature, threshold, split_impurity = find_best_split(
            X,
            y,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            rng=self.rng,
            criterion=self.criterion
        )

        if self.criterion == "gini":
            parent_impurity = 1 - np.sum(leaf["probabilities"] ** 2)
        else:
            probabilities = leaf["probabilities"][counts != 0]
            parent_impurity = -np.sum(probabilities * np.log2(probabilities))

        # Stop if no useful split exists
        if (
            feature is None
            or split_impurity >= parent_impurity - 1e-12
        ):

            return leaf

        # Divide data into two groups
        left_mask = X[:, feature].astype(np.float64, copy=False) <= threshold
        right_mask = ~left_mask

        # Recursively build child nodes
        left_tree = self._grow_tree(
            X[left_mask],
            y[left_mask],
            depth + 1
        )

        right_tree = self._grow_tree(
            X[right_mask],
            y[right_mask],
            depth + 1
        )

        # Return decision node
        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree
        }

    def _fill_predictions(self, X, node, rows, output, key):
        if rows.size == 0:
            return
        if "class" in node:
            output[rows] = node[key]
            return

        values = X[rows, node["feature"]].astype(np.float64, copy=False)
        left = values <= node["threshold"]
        self._fill_predictions(X, node["left"], rows[left], output, key)
        self._fill_predictions(X, node["right"], rows[~left], output, key)

    def predict(self, X):
        if self.root is None:
            raise ValueError("Train the tree before predicting.")

        output = np.empty(len(X), dtype=int)
        self._fill_predictions(X, self.root, np.arange(len(X)), output, "class")
        return output

    def predict_proba(self, X):
        """Return normalized leaf class counts in classes_ order."""
        if self.root is None:
            raise ValueError("Train the tree before predicting.")

        output = np.empty((len(X), len(self.classes_)))
        self._fill_predictions(
            X, self.root, np.arange(len(X)), output, "probabilities"
        )
        return output
