"""Audio quality probing (header-level) and signal-level quality metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def probe_audio(path: Path | str) -> dict:
    """Read only the WAV header: duration, sample rate, channels. Never raises."""
    path = Path(path)
    out = {"filename": path.name, "path": str(path), "ok": False, "duration": np.nan,
           "sample_rate": np.nan, "channels": np.nan, "frames": np.nan, "error": ""}
    try:
        info = sf.info(str(path))
        out.update(ok=info.frames > 0, duration=float(info.duration), sample_rate=int(info.samplerate),
                   channels=int(info.channels), frames=int(info.frames))
        if info.frames <= 0:
            out["error"] = "zero frames"
    except Exception as exc:  # corrupt / not a WAV / unreadable
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def silence_ratio(y: np.ndarray, top_db: float = 30.0, frame_length: int = 1024, hop_length: int = 256) -> float:
    """Fraction of the signal that is below ``top_db`` of the peak (1.0 for all-silent audio)."""
    import librosa

    if y.size == 0 or not np.any(np.abs(y) > 0):
        return 1.0
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    voiced = int(sum(e - s for s, e in intervals))
    return float(1.0 - voiced / len(y))


def is_effectively_silent(y: np.ndarray, rms_threshold: float = 1e-4) -> bool:
    return y.size == 0 or float(np.sqrt(np.mean(np.square(y)))) < rms_threshold


def clipping_ratio(y: np.ndarray, threshold: float = 0.999) -> float:
    if y.size == 0:
        return 0.0
    return float(np.mean(np.abs(y) >= threshold))
