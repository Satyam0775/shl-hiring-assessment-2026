"""Reusable, non-destructive audio preprocessing (originals are never modified)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class AudioLoadError(RuntimeError):
    """Raised when audio cannot be decoded or is unusable."""


def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    return y.mean(axis=0) if y.ndim == 2 else y


def remove_dc(y: np.ndarray) -> np.ndarray:
    return y - float(np.mean(y)) if y.size else y


def peak_normalize(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale to unit peak; all-zero input is returned unchanged (safe)."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return y if peak < eps else (y / peak).astype(np.float32)


def load_audio(path: Path | str, sr: int = 16000, min_duration: float = 0.0) -> tuple[np.ndarray, int]:
    """Load as mono float32 at ``sr`` Hz. Raises AudioLoadError for corrupt/too-short audio."""
    import librosa

    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"audio file not found: {path}")
    try:
        y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        raise AudioLoadError(f"{type(exc).__name__}: {exc}") from exc
    if y.size == 0:
        raise AudioLoadError("decoded audio is empty")
    if len(y) / sr_out < min_duration:
        raise AudioLoadError(f"audio shorter than {min_duration}s ({len(y) / sr_out:.2f}s)")
    return y.astype(np.float32), int(sr_out)
