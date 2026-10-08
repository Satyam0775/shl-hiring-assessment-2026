"""Simple averaging ensemble (EXP-07). Used only if CV proves it helps."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from src.evaluation.cross_validation import CVResult, result_from_oof

BASELINES = ("mean_baseline", "median_baseline")


class AveragingEnsemble(BaseEstimator, RegressorMixin):
    """Fits each member pipeline on the same data and averages their predictions."""

    def __init__(self, members=None):
        self.members = members  # list of (name, unfitted pipeline)

    def fit(self, X, y):
        self.fitted_ = [(n, clone(m).fit(X, y)) for n, m in self.members]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for _, m in self.fitted_], axis=0)


def oof_column(experiment_id: str, model: str) -> str:
    return f"{experiment_id}__{model}"


def top_k_members(log: pd.DataFrame, k: int = 3) -> list[dict]:
    """Best ``k`` distinct (experiment, model) pairs by CV RMSE, excluding baselines and ensembles."""
    d = log[~log["model"].isin(BASELINES) & ~log["model"].str.startswith("ensemble")].copy()
    d = d.sort_values(["cv_rmse_mean", "cv_pearson_mean"], ascending=[True, False]).head(k)
    return d[["experiment_id", "model", "feature_set"]].to_dict("records")


def evaluate_oof_ensemble(oof_table: pd.DataFrame, members: list[dict], y, folds) -> CVResult:
    cols = [oof_column(m["experiment_id"], m["model"]) for m in members]
    absent = [c for c in cols if c not in oof_table.columns]
    if absent:
        raise KeyError(f"OOF predictions missing for {absent}; run those experiments first")
    return result_from_oof(y, oof_table[cols].mean(axis=1).to_numpy(), folds)


def members_to_json(members: list[dict]) -> str:
    return json.dumps({"members": members})
