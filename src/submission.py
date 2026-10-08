"""Submission validation and versioned writing. Never overwrites the official sample file."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class SubmissionError(RuntimeError):
    """Raised when a submission would not be valid for the official format."""


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame, min_score: float = 0.0,
                        max_score: float = 5.0, reorder: bool = True) -> pd.DataFrame:
    """Validate against the official sample file and return the submission in the sample's row order.

    Checks: schema (names + order), duplicates, NaNs, numeric dtype, filename set equality, count, range.
    Raises SubmissionError listing every problem. Nothing is invented, dropped or duplicated.
    """
    problems: list[str] = []
    if list(sample.columns) != list(sub.columns):
        problems.append(f"columns {list(sub.columns)} != expected {list(sample.columns)} (names and order must match)")
    if problems:
        raise SubmissionError("; ".join(problems))
    id_col, val_col = sample.columns[0], sample.columns[1]
    if sub[id_col].duplicated().any():
        problems.append(f"{int(sub[id_col].duplicated().sum())} duplicate filenames in submission")
    if sample[id_col].duplicated().any():
        problems.append("sample_submission itself contains duplicate filenames")
    if sub.isna().any().any():
        problems.append(f"{int(sub.isna().sum().sum())} missing values in submission")
    pred = pd.to_numeric(sub[val_col], errors="coerce")
    if pred.isna().any() and not sub[val_col].isna().any():
        problems.append("non-numeric predictions present")
    if len(sub) != len(sample):
        problems.append(f"row count {len(sub)} != sample_submission row count {len(sample)}")
    s_names, p_names = set(sample[id_col]), set(sub[id_col])
    if s_names != p_names:
        problems.append(f"filename sets differ: {len(p_names - s_names)} predicted names not in sample_submission, "
                        f"{len(s_names - p_names)} sample_submission names not predicted")
    if not pred.isna().any() and ((pred < min_score) | (pred > max_score)).any():
        problems.append(f"predictions outside [{min_score}, {max_score}]")
    if problems:
        raise SubmissionError("Submission invalid:\n - " + "\n - ".join(problems))
    out = sub.copy()
    out[val_col] = pred.astype(float)
    if reorder:
        out = out.set_index(id_col).loc[sample[id_col]].reset_index()[list(sample.columns)]
    return out


def submission_path(directory: Path | str, version: Optional[int] = None, name: str = "final") -> Path:
    d = Path(directory)
    if version is None:
        return d / f"submission_{name}.csv"
    return d / f"submission_v{version:02d}_{name}.csv"


def write_submission(sub: pd.DataFrame, path: Path | str, protected: Optional[Path | str] = None,
                     overwrite: bool = False) -> Path:
    path = Path(path)
    if protected is not None and path.resolve() == Path(protected).resolve():
        raise SubmissionError(f"refusing to overwrite the official file {protected}")
    if path.exists() and not overwrite:
        raise SubmissionError(f"{path} already exists; pass --overwrite or choose a new version")
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")
    return path
