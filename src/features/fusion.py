"""Feature-table merging. Column prefixes define feature groups."""
from __future__ import annotations

from typing import Optional

import pandas as pd

GROUP_PREFIXES = {"audio": ("aud_",), "text": ("txt_", "ling_"), "embeddings": ("emb_",)}
ALL_GROUPS = tuple(GROUP_PREFIXES)


def group_columns(columns, groups) -> list[str]:
    prefixes = tuple(p for g in groups for p in GROUP_PREFIXES[g])
    return [c for c in columns if c.startswith(prefixes)] if prefixes else []


def _feature_only(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["filename"] + [c for c in df.columns if c.startswith(tuple(p for v in GROUP_PREFIXES.values() for p in v))]
    return df[keep]


def merge_feature_tables(manifest: pd.DataFrame, audio: Optional[pd.DataFrame] = None,
                         text: Optional[pd.DataFrame] = None, emb: Optional[pd.DataFrame] = None,
                         extra: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Left-join feature tables onto ``manifest`` (its row order and row count are preserved).

    Raises if any table has duplicate filenames or if the join changes the row count.
    """
    out = manifest.copy().reset_index(drop=True)
    n = len(out)
    for name, tbl in (("audio", audio), ("text", text), ("embeddings", emb), ("extra", extra)):
        if tbl is None:
            continue
        if tbl["filename"].duplicated().any():
            raise ValueError(f"{name} feature table has duplicate filenames")
        out = out.merge(_feature_only(tbl) if name != "extra" else tbl, on="filename", how="left", validate="one_to_one")
        if len(out) != n:
            raise ValueError(f"merging {name} features changed the row count ({n} -> {len(out)})")
    return out


def rows_missing_features(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Manifest rows for which every column of some requested group is NaN (feature extraction never ran / failed)."""
    bad = pd.Series(False, index=df.index)
    for g in groups:
        cols = group_columns(df.columns, [g])
        if not cols:
            bad |= True
        else:
            bad |= df[cols].isna().all(axis=1)
    return df[bad]
