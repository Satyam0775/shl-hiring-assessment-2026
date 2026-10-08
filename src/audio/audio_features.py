"""Acoustic feature extraction (all columns prefixed ``aud_``).

Only features with a clear definition are used. Pitch is a rough YIN estimate
and the speaking-rate proxy is an onset rate; both are proxies, not ground truth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.audio.preprocessing import AudioLoadError, load_audio, peak_normalize, remove_dc

N_FFT = 1024
HOP = 256
N_MFCC = 13


def _stats(prefix: str, x: np.ndarray, out: dict) -> None:
    x = np.asarray(x, dtype=float)
    out[f"{prefix}_mean"] = float(np.mean(x)) if x.size else np.nan
    out[f"{prefix}_std"] = float(np.std(x)) if x.size else np.nan


def extract_audio_features(y: np.ndarray, sr: int, silence_top_db: float = 30.0,
                           pause_min_sec: float = 0.3) -> dict[str, float]:
    """Compute the acoustic feature dict for one mono waveform."""
    import librosa

    f: dict[str, float] = {}
    y = remove_dc(np.asarray(y, dtype=np.float32))
    if len(y) < N_FFT:                      # guard for very short clips
        y = np.pad(y, (0, N_FFT - len(y)))
    duration = len(y) / sr
    f["aud_duration"] = float(duration)

    # --- level features on the un-normalised signal
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    _stats("aud_rms", rms, f)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    _stats("aud_zcr", zcr, f)

    # --- spectral features on the peak-normalised signal (removes recording-gain effects)
    yn = peak_normalize(y)
    for name, fn in (("centroid", librosa.feature.spectral_centroid),
                     ("bandwidth", librosa.feature.spectral_bandwidth),
                     ("rolloff", librosa.feature.spectral_rolloff)):
        _stats(f"aud_{name}", fn(y=yn, sr=sr, n_fft=N_FFT, hop_length=HOP)[0], f)
    _stats("aud_flatness", librosa.feature.spectral_flatness(y=yn, n_fft=N_FFT, hop_length=HOP)[0], f)
    contrast = librosa.feature.spectral_contrast(y=yn, sr=sr, n_fft=N_FFT, hop_length=HOP)
    _stats("aud_contrast", contrast.mean(axis=0), f)

    mfcc = librosa.feature.mfcc(y=yn, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    delta = librosa.feature.delta(mfcc) if mfcc.shape[1] >= 9 else np.zeros_like(mfcc)
    for i in range(N_MFCC):
        _stats(f"aud_mfcc{i + 1:02d}", mfcc[i], f)
        _stats(f"aud_dmfcc{i + 1:02d}", delta[i], f)

    # --- speech activity / pauses
    if float(np.max(np.abs(y))) < 1e-4:   # effectively silent: librosa's relative threshold would call it all speech
        intervals = np.zeros((0, 2), dtype=int)
    else:
        intervals = librosa.effects.split(yn, top_db=silence_top_db, frame_length=N_FFT, hop_length=HOP)
    speech_sec = float(sum(e - s for s, e in intervals) / sr)
    f["aud_speech_ratio"] = speech_sec / duration if duration else np.nan
    f["aud_silence_ratio"] = 1.0 - f["aud_speech_ratio"]
    gaps = np.array([(intervals[i + 1][0] - intervals[i][1]) / sr for i in range(len(intervals) - 1)], dtype=float)
    pauses = gaps[gaps >= pause_min_sec] if gaps.size else gaps
    f["aud_n_pauses"] = float(pauses.size)
    f["aud_pauses_per_min"] = float(pauses.size / (duration / 60.0)) if duration else np.nan
    f["aud_pause_mean"] = float(pauses.mean()) if pauses.size else 0.0
    f["aud_pause_max"] = float(pauses.max()) if pauses.size else 0.0
    f["aud_pause_total"] = float(pauses.sum()) if pauses.size else 0.0
    f["aud_n_speech_segments"] = float(len(intervals))

    # --- speaking-rate proxy: acoustic onsets per second of speech
    if speech_sec > 0.5:
        onsets = librosa.onset.onset_detect(y=yn, sr=sr, hop_length=HOP, units="frames")
        f["aud_onset_rate"] = float(len(onsets) / speech_sec)
    else:
        f["aud_onset_rate"] = 0.0

    # --- rough pitch (YIN) on energetic frames
    try:
        f0 = librosa.yin(yn, fmin=65, fmax=400, sr=sr, frame_length=N_FFT * 2, hop_length=HOP)
        rms_n = librosa.feature.rms(y=yn, frame_length=N_FFT * 2, hop_length=HOP)[0]
        m = min(len(f0), len(rms_n))
        voiced = (rms_n[:m] > 0.3 * rms_n[:m].mean()) & (f0[:m] < 399) & (f0[:m] > 66)
        v = f0[:m][voiced]
        if v.size > 10:
            f["aud_pitch_mean"], f["aud_pitch_std"] = float(v.mean()), float(v.std())
            f["aud_pitch_range"] = float(np.percentile(v, 95) - np.percentile(v, 5))
        else:
            f["aud_pitch_mean"] = f["aud_pitch_std"] = f["aud_pitch_range"] = np.nan
    except Exception:
        f["aud_pitch_mean"] = f["aud_pitch_std"] = f["aud_pitch_range"] = np.nan
    return f


def extract_features_for_file(path: Path | str, cfg_audio: Optional[dict] = None) -> dict:
    """Load + featurise one file. Never raises: failures are recorded in status/error_message."""
    cfg_audio = cfg_audio or {}
    path = Path(path)
    row: dict = {"filename": path.name, "status": "ok", "error_message": ""}
    try:
        y, sr = load_audio(path, sr=int(cfg_audio.get("sample_rate", 16000)),
                           min_duration=float(cfg_audio.get("min_duration_sec", 0.0)))
        row.update(extract_audio_features(y, sr, float(cfg_audio.get("silence_top_db", 30)),
                                          float(cfg_audio.get("pause_min_sec", 0.3))))
    except AudioLoadError as exc:
        row.update(status="error", error_message=str(exc))
    except Exception as exc:
        row.update(status="error", error_message=f"{type(exc).__name__}: {exc}")
    return row
