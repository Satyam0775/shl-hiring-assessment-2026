"""Competition metrics. RMSE: lower is better. Pearson: higher is better."""
from __future__ import annotations

import numpy as np


def _validate(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("empty input arrays")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("NaN values in y_true or y_pred")
    if np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("infinite values in y_true or y_pred")
    return a, b


def rmse(y_true, y_pred) -> float:
    a, b = _validate(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def pearson(y_true, y_pred) -> float:
    """Pearson r. If either input is constant (e.g. a mean baseline) r is undefined; we return 0.0
    (no linear relationship) so that aggregate statistics stay finite. Needs >= 2 points."""
    a, b = _validate(y_true, y_pred)
    if a.size < 2:
        raise ValueError("Pearson correlation needs at least 2 samples")
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def clip_predictions(pred, low: float = 0.0, high: float = 5.0) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float), low, high)


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {"rmse": rmse(y_true, y_pred), "pearson": pearson(y_true, y_pred)}
