"""Local ASR with faster-whisper, with resumable caching.

The expensive part (running Whisper) is isolated behind a ``transcribe_fn``
callable so the caching/resume logic is unit-testable without a model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from src.utils import read_cache, upsert_rows, write_csv_atomic

TRANSCRIPT_COLUMNS = ["filename", "transcript", "duration", "status", "error_message", "asr_model"]


def detect_device() -> str:
    """'cuda' if an NVIDIA GPU is visible to CTranslate2, else 'cpu'. GPU is optional."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type and compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


class WhisperTranscriber:
    """Thin lazy wrapper around faster-whisper with model fallback."""

    def __init__(self, model: str = "small.en", fallback_model: Optional[str] = "base.en",
                 language: str = "en", compute_type: str = "auto", beam_size: int = 5,
                 vad_filter: bool = False, device: str = "auto"):
        self.requested_model, self.fallback_model = model, fallback_model
        self.language, self.beam_size, self.vad_filter = language, beam_size, vad_filter
        self.device = detect_device() if device == "auto" else device
        self.compute_type = resolve_compute_type(compute_type, self.device)
        self.model_name: Optional[str] = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        for name in [self.requested_model, self.fallback_model]:
            if not name:
                continue
            try:
                self._model = WhisperModel(name, device=self.device, compute_type=self.compute_type)
                self.model_name = name
                if name != self.requested_model:
                    print(f"[asr] WARNING: using fallback model '{name}' (requested '{self.requested_model}' failed to load)")
                return
            except Exception as exc:
                print(f"[asr] could not load '{name}': {type(exc).__name__}: {exc}")
        raise RuntimeError("No Whisper model could be loaded (check internet access for first download, or the model names).")

    def transcribe_file(self, path: Path | str) -> tuple[str, float]:
        """Return (transcript, audio_duration_sec). Raises on failure."""
        self._load()
        segments, info = self._model.transcribe(
            str(path), language=self.language, beam_size=self.beam_size, vad_filter=self.vad_filter,
            condition_on_previous_text=False, temperature=0.0)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, float(info.duration)

    @property
    def asr_model(self) -> str:
        return self.model_name or self.requested_model


def transcribe_manifest(filenames: Iterable[str], audio_dir: Path | str, out_csv: Path | str,
                        transcribe_fn: Callable[[Path], tuple[str, float]], asr_model: str,
                        force: bool = False, retry_failed: bool = False, save_every: int = 10,
                        progress: bool = True) -> pd.DataFrame:
    """Transcribe ``filenames`` with caching/resume.

    * already-OK rows are skipped (unless ``force``)
    * previously failed rows are skipped unless ``retry_failed``
    * progress is flushed to disk every ``save_every`` files
    * one failing file never aborts the run
    """
    from tqdm import tqdm

    audio_dir, out_csv = Path(audio_dir), Path(out_csv)
    wanted = list(dict.fromkeys(filenames))
    cache = None if force else read_cache(out_csv)
    if cache is not None:
        cache = cache.copy()
        cache["transcript"] = cache["transcript"].fillna("")
        status = cache.set_index("filename")["status"].to_dict()
    else:
        status = {}

    def needs_work(fn: str) -> bool:
        s = status.get(fn)
        return s is None or (s != "ok" and retry_failed)

    todo = [fn for fn in wanted if needs_work(fn)]
    skipped_failed = [fn for fn in wanted if status.get(fn) not in (None, "ok") and not retry_failed]
    print(f"[asr] {len(wanted)} requested | {len(wanted) - len(todo) - len(skipped_failed)} cached OK | "
          f"{len(todo)} to transcribe | {len(skipped_failed)} previously failed (use retry to re-run)")

    results: list[dict] = []
    current = cache

    def flush():
        nonlocal current, results
        if results:
            current = upsert_rows(current, pd.DataFrame(results, columns=TRANSCRIPT_COLUMNS))
            write_csv_atomic(current, out_csv)
            results = []

    for i, fn in enumerate(tqdm(todo, desc="ASR", disable=not progress), start=1):
        rec = {"filename": fn, "transcript": "", "duration": float("nan"), "status": "ok",
               "error_message": "", "asr_model": asr_model}
        try:
            path = audio_dir / fn
            if not path.exists():
                raise FileNotFoundError(f"audio file not found: {path}")
            rec["transcript"], rec["duration"] = transcribe_fn(path)
        except Exception as exc:
            rec["status"], rec["error_message"] = "error", f"{type(exc).__name__}: {exc}"
            print(f"[asr] FAILED {fn}: {rec['error_message']}")
        results.append(rec)
        if i % max(1, save_every) == 0:
            flush()
    flush()
    final = current if current is not None else pd.DataFrame(columns=TRANSCRIPT_COLUMNS)
    return final
