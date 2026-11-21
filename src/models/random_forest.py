"""Random Forest, from scratch, built on our DecisionTree.

Each tree is trained on a bootstrap sample and looks at a random subset of features at
every split; the forest averages the trees' votes (hard) or leaf probabilities (soft).
"""
import numpy as np

from .base import BaseClassifier
from .decision_tree import DecisionTree


class RandomForest(BaseClassifier):
    PARAMS = ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
              "max_features", "criterion", "voting", "random_state")

    def __init__(
        self,
        n_estimators=10,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        voting="hard",
        verbose=True,
        criterion="gini"
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion

        if voting not in {"hard", "soft"}:
            raise ValueError("voting must be 'hard' or 'soft'.")
        self.voting = voting
        self.verbose = verbose
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Determine how many features each split examines
        if self.max_features == "sqrt":
            n_selected_features = max(
                1, int(np.sqrt(n_features))
            )
        elif self.max_features is None:
            n_selected_features = None
        else:
            n_selected_features = int(self.max_features)

        # Reset trees and random generator
        self.trees = []
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)

        for i in range(self.n_estimators):

            # Bootstrap sample: draw with replacement
            sample_indices = rng.choice(
                n_samples,
                size=n_samples,
                replace=True
            )

            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # Create an independent Decision Tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=n_selected_features,
                random_state=self.random_state + i,
                criterion=self.criterion
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

            if self.verbose:
                print(f"Trained tree {i + 1}/{self.n_estimators}", flush=True)

        return self

    def predict(self, X):
        scores = self.predict_proba(X)
        return self.classes_[scores.argmax(axis=1)]

    def predict_proba(self, X):
        """Average hard votes or leaf class probabilities in classes_ order."""
        if not self.trees:
            raise ValueError("Train the forest before predicting.")

        scores = np.zeros((len(X), len(self.classes_)))
        rows = np.arange(len(X))
        for tree in self.trees:
            if self.voting == "soft":
                columns = np.searchsorted(self.classes_, tree.classes_)
                scores[:, columns] += tree.predict_proba(X)
            else:
                columns = np.searchsorted(self.classes_, tree.predict(X))
                scores[rows, columns] += 1
        return scores / len(self.trees)
