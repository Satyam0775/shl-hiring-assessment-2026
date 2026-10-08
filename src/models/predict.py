"""Test-time inference: load model, check feature columns, predict, clip to [0, 5]."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import clip_predictions


def load_model(path: Path | str):
    bundle = joblib.load(path)
    return bundle["model"], bundle["meta"]


def predict_scores(model, features: pd.DataFrame, min_score: float = 0.0, max_score: float = 5.0) -> pd.DataFrame:
    """Return DataFrame[filename, label] with predictions clipped to the valid range."""
    if features["filename"].duplicated().any():
        raise ValueError("duplicate filenames in test features")
    raw = np.asarray(model.predict(features), dtype=float)
    if np.isnan(raw).any():
        raise ValueError("model produced NaN predictions")
    return pd.DataFrame({"filename": features["filename"].to_numpy(), "label": clip_predictions(raw, min_score, max_score)})
