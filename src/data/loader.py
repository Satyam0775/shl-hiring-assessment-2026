"""Metadata loading and path resolution. Never modifies or rewrites the competition CSVs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import DataPaths

REQUIRED_COLUMNS = ["filename", "label"]
AUDIO_EXTENSIONS = (".wav",)


class SchemaError(ValueError):
    """Raised when a CSV does not have the expected columns."""


@dataclass
class Metadata:
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame


def check_schema(df: pd.DataFrame, name: str, required: list[str] | None = None) -> None:
    required = required or REQUIRED_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}; found {list(df.columns)}")


def read_metadata_csv(path: Path | str, name: str = "csv") -> pd.DataFrame:
    """Read a metadata CSV as-is (utf-8-sig tolerates a Windows BOM)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found at: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"filename": str})
    check_schema(df, name)
    return df


def load_all_metadata(paths: DataPaths) -> Metadata:
    missing = paths.missing_csvs()
    if missing:
        raise FileNotFoundError(
            "Expected dataset layout: <data-root>/{train.csv,test.csv,sample_submission.csv,train/,test/}. "
            f"Missing: {[str(m) for m in missing]}"
        )
    return Metadata(
        train=read_metadata_csv(paths.train_csv, "train.csv"),
        test=read_metadata_csv(paths.test_csv, "test.csv"),
        sample=read_metadata_csv(paths.sample_submission_csv, "sample_submission.csv"),
    )


def get_manifest(meta: Metadata, manifest: str = "test") -> pd.DataFrame:
    """Which filename list drives test-time processing: 'test' (test.csv) or 'sample_submission'."""
    if manifest == "test":
        return meta.test
    if manifest in ("sample_submission", "sample"):
        return meta.sample
    raise ValueError("manifest must be 'test' or 'sample_submission'")


def resolve_audio_path(paths: DataPaths, split: str, filename: str) -> Path:
    """Join audio dir and filename; refuses anything that is not a bare file name."""
    if Path(filename).name != filename or not filename:
        raise ValueError(f"Unsafe filename in metadata: {filename!r}")
    return paths.audio_dir(split) / filename


def list_audio_files(directory: Path | str) -> list[str]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
