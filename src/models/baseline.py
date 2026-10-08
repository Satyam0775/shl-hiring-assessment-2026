"""Constant baselines (EXP-00 mean, EXP-01 median). They ignore features entirely."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class ConstantBaseline(BaseEstimator, RegressorMixin):
    """Predict the training-fold mean or median for every sample."""

    def __init__(self, strategy: str = "mean"):
        self.strategy = strategy

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if self.strategy == "mean":
            self.constant_ = float(np.mean(y))
        elif self.strategy == "median":
            self.constant_ = float(np.median(y))
        else:
            raise ValueError("strategy must be 'mean' or 'median'")
        return self

    def predict(self, X):
        return np.full(len(X), self.constant_, dtype=float)
