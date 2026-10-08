"""Stratified K-fold cross-validation for a continuous target (leakage-safe)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import clip_predictions, pearson, rmse


def make_folds(y, n_splits: int = 5, seed: int = 42, n_bins: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratify on equal-frequency bins of the score (rank-based, so tied labels never break binning)."""
    y = np.asarray(y, dtype=float)
    bins = pd.qcut(pd.Series(y).rank(method="first"), q=n_bins, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), bins)]


def fold_id_vector(folds, n: int) -> np.ndarray:
    ids = np.full(n, -1, dtype=int)
    for k, (_, va) in enumerate(folds):
        ids[va] = k
    return ids


@dataclass
class CVResult:
    fold_rmse: list[float]
    fold_pearson: list[float]
    oof: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def rmse_mean(self): return float(np.mean(self.fold_rmse))
    @property
    def rmse_std(self): return float(np.std(self.fold_rmse, ddof=1)) if len(self.fold_rmse) > 1 else 0.0
    @property
    def pearson_mean(self): return float(np.mean(self.fold_pearson))
    @property
    def pearson_std(self): return float(np.std(self.fold_pearson, ddof=1)) if len(self.fold_pearson) > 1 else 0.0

    @property
    def oof_rmse(self): return rmse(self._y, self.oof)
    @property
    def oof_pearson(self): return pearson(self._y, self.oof)

    _y: np.ndarray = field(default=None, repr=False)


def result_from_oof(y, oof, folds) -> CVResult:
    """Per-fold metrics from an existing out-of-fold prediction vector (used for ensembles)."""
    y, oof = np.asarray(y, float), np.asarray(oof, float)
    fr = [rmse(y[va], oof[va]) for _, va in folds]
    fp = [pearson(y[va], oof[va]) for _, va in folds]
    return CVResult(fr, fp, oof, _y=y)


def cross_validate(estimator, X: pd.DataFrame, y, folds, clip: tuple[float, float] = (0.0, 5.0)) -> CVResult:
    """Clone + fit ``estimator`` on each training fold ONLY, predict the held-out fold, clip to the score range.

    The estimator should be a Pipeline whose preprocessing (imputer / PCA) lives inside it,
    so that every data-dependent step is refitted per fold.
    """
    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan)
    for tr, va in folds:
        model = clone(estimator)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = clip_predictions(model.predict(X.iloc[va]), *clip)
    if np.isnan(oof).any():
        raise RuntimeError("some rows never appeared in a validation fold")
    return result_from_oof(y, oof, folds)
