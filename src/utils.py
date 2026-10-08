"""Small shared helpers: atomic CSV writes and keyed-cache merging."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


def write_csv_atomic(df: pd.DataFrame, path: os.PathLike | str) -> Path:
    """Write a UTF-8 CSV atomically (temp file + replace) so interrupted runs never corrupt caches."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_cache(path: os.PathLike | str) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8", keep_default_na=True)


def upsert_rows(existing: pd.DataFrame | None, new: pd.DataFrame, key: str = "filename") -> pd.DataFrame:
    """Merge ``new`` into ``existing``; rows in ``new`` win on duplicate keys."""
    if existing is None or existing.empty:
        return new.reset_index(drop=True)
    keep = existing[~existing[key].isin(new[key])]
    return pd.concat([keep, new], ignore_index=True)


def chunked(seq: Iterable, n: int):
    buf = []
    for item in seq:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
