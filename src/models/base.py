"""Common interface for every classifier in this project."""
import numpy as np


class BaseClassifier:
    """Subclass this. Implement fit() and predict(); predict_proba() if it's natural.

    Store hyperparameters as attributes in __init__ and list their names in PARAMS,
    so get_params() can log them automatically.
    """

    PARAMS: tuple[str, ...] = ()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no predict_proba")

    def get_params(self) -> dict:
        return {p: getattr(self, p) for p in self.PARAMS}

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"
