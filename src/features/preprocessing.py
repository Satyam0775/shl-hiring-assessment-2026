"""Fold-safe feature preprocessing.

``GroupFeatureBuilder`` is a scikit-learn transformer that is fitted ONLY on the
training rows it is given. Inside ``cross_validate`` it is cloned and refitted
per fold, so the imputer medians and the PCA basis never see validation data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from src.features.fusion import group_columns


class GroupFeatureBuilder(BaseEstimator, TransformerMixin):
    """Select feature groups, median-impute tabular columns, optionally PCA the embeddings."""

    def __init__(self, groups=("audio", "text"), use_pca=True, pca_components=50, random_state=42):
        self.groups = groups
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("GroupFeatureBuilder expects a pandas DataFrame with prefixed feature columns")
        groups = list(self.groups)
        self.tab_cols_ = group_columns(X.columns, [g for g in groups if g in ("audio", "text")])
        self.emb_cols_ = group_columns(X.columns, ["embeddings"]) if "embeddings" in groups else []
        if not self.tab_cols_ and not self.emb_cols_:
            raise ValueError(f"No feature columns found for groups {groups}")
        names: list[str] = []
        if self.tab_cols_:
            self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
            self.imputer_.fit(self._tab(X))
            names += self.tab_cols_
        self.pca_ = None
        if self.emb_cols_:
            E = X[self.emb_cols_].to_numpy(dtype=float)
            if self.use_pca:
                k = int(min(self.pca_components, E.shape[0] - 1, E.shape[1]))
                self.pca_ = PCA(n_components=k, random_state=self.random_state).fit(E)
                names += [f"emb_pc{i:02d}" for i in range(k)]
            else:
                names += self.emb_cols_
        self.feature_names_out_ = names
        return self

    def _tab(self, X):
        return X[self.tab_cols_].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

    def transform(self, X: pd.DataFrame):
        missing = [c for c in self.tab_cols_ + self.emb_cols_ if c not in X.columns]
        if missing:
            raise KeyError(f"{len(missing)} expected feature columns are missing, e.g. {missing[:5]}")
        parts = []
        if self.tab_cols_:
            parts.append(self.imputer_.transform(self._tab(X)))
        if self.emb_cols_:
            E = X[self.emb_cols_].to_numpy(dtype=float)
            parts.append(self.pca_.transform(E) if self.pca_ is not None else E)
        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_)
