"""Dataset integrity audit.

Design rule: the audit REPORTS inconsistencies. It never silently fixes,
drops, duplicates or invents rows.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.audio.quality import probe_audio
from src.config import DataPaths
from src.data.loader import REQUIRED_COLUMNS, list_audio_files

FILENAME_PATTERN = re.compile(r"^audio_(\d+)\.wav$")
MAX_EXAMPLES = 10


class MetadataInconsistencyError(RuntimeError):
    """Raised when test.csv and sample_submission.csv (or other metadata) disagree."""


@dataclass
class Finding:
    level: str   # "error" | "warning" | "info"
    code: str
    message: str


@dataclass
class AuditReport:
    stats: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    inventory: Optional[pd.DataFrame] = None  # per-file audio probe results (not serialised)

    def add(self, level: str, code: str, message: str) -> None:
        self.findings.append(Finding(level, code, message))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def to_dict(self) -> dict:
        return {"stats": self.stats, "findings": [f.__dict__ for f in self.findings]}

    def to_json(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=_json_default), encoding="utf-8")

    def to_markdown(self) -> str:
        lines = ["# Dataset audit report", ""]
        lines.append(f"**Errors:** {len(self.errors)}  |  **Warnings:** {len(self.warnings)}")
        lines.append("")
        lines.append("## Findings")
        if not self.findings:
            lines.append("- none")
        for f in self.findings:
            lines.append(f"- **[{f.level.upper()}] {f.code}** - {f.message}")
        lines.append("")
        lines.append("## Statistics")
        lines.append("```json")
        lines.append(json.dumps(self.stats, indent=2, default=_json_default))
        lines.append("```")
        return "\n".join(lines)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (set, tuple)):
        return sorted(o)
    return str(o)


def _examples(items) -> list[str]:
    return sorted(items)[:MAX_EXAMPLES]


def _filename_number(names: pd.Series) -> pd.Series:
    return pd.to_numeric(names.astype(str).str.extract(FILENAME_PATTERN)[0], errors="coerce")


def _frame_stats(df: pd.DataFrame, name: str, report: AuditReport) -> dict:
    s: dict = {"rows": int(len(df)), "columns": list(df.columns)}
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.add("error", f"{name.upper()}_SCHEMA", f"{name}: missing columns {missing_cols}")
        return s
    s["missing_values"] = {c: int(v) for c, v in df.isna().sum().items()}
    s["duplicate_filenames"] = int(df["filename"].duplicated().sum())
    nums = _filename_number(df["filename"])
    s["filename_pattern_match"] = int(nums.notna().sum())
    s["filename_number_min"] = None if nums.isna().all() else int(nums.min())
    s["filename_number_max"] = None if nums.isna().all() else int(nums.max())
    if int(df.isna().sum().sum()) > 0:
        report.add("error", f"{name.upper()}_MISSING_VALUES", f"{name}: contains missing values {s['missing_values']}")
    if s["duplicate_filenames"] > 0:
        report.add("error", f"{name.upper()}_DUPLICATES", f"{name}: {s['duplicate_filenames']} duplicate filenames")
    if s["filename_pattern_match"] != len(df):
        report.add("warning", f"{name.upper()}_FILENAME_PATTERN",
                   f"{name}: {len(df) - s['filename_pattern_match']} filenames do not match 'audio_<n>.wav'")
    return s


def audit_metadata(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame,
                   expected_train: Optional[int] = 769, expected_test: Optional[int] = 216,
                   score_range: tuple[float, float] = (0.0, 5.0)) -> AuditReport:
    """CSV-only audit (no audio needed)."""
    r = AuditReport()
    r.stats["train"] = _frame_stats(train, "train", r)
    r.stats["test"] = _frame_stats(test, "test", r)
    r.stats["sample_submission"] = _frame_stats(sample, "sample_submission", r)
    if r.has_errors and any(f.code.endswith("_SCHEMA") for f in r.errors):
        return r

    if expected_train is not None and len(train) != expected_train:
        r.add("warning", "TRAIN_ROW_COUNT", f"train.csv has {len(train)} rows; assignment says {expected_train}")
    if expected_test is not None and len(test) != expected_test:
        r.add("warning", "TEST_ROW_COUNT", f"test.csv has {len(test)} rows; assignment says {expected_test}")

    # ---- label checks
    lab = pd.to_numeric(train["label"], errors="coerce")
    lo, hi = score_range
    r.stats["train_label"] = {
        "min": None if lab.isna().all() else float(lab.min()),
        "max": None if lab.isna().all() else float(lab.max()),
        "mean": None if lab.isna().all() else float(lab.mean()),
        "std": None if lab.isna().all() else float(lab.std()),
        "value_counts": {str(k): int(v) for k, v in lab.value_counts().sort_index().items()},
    }
    if lab.isna().any():
        r.add("error", "TRAIN_LABEL_NON_NUMERIC", f"{int(lab.isna().sum())} train labels are non-numeric/missing")
    elif lab.min() < lo or lab.max() > hi:
        r.add("error", "TRAIN_LABEL_RANGE", f"train labels outside [{lo}, {hi}]: min={lab.min()}, max={lab.max()}")
    r.stats["test_label_values"] = sorted({str(v) for v in test["label"].unique()})
    r.stats["sample_label_values"] = sorted({str(v) for v in sample["label"].unique()})
    if not (pd.to_numeric(test["label"], errors="coerce") == -1).all():
        r.add("warning", "TEST_LABELS_NOT_SENTINEL", "test.csv labels are not all -1 (they are never used as ground truth anyway)")

    # ---- filename set relations
    tr, te, ss = set(train["filename"]), set(test["filename"]), set(sample["filename"])
    r.stats["overlap"] = {
        "train_and_test": len(tr & te), "test_and_sample": len(te & ss), "train_and_sample": len(tr & ss),
        "test_only_vs_sample": len(te - ss), "sample_only_vs_test": len(ss - te),
        "test_only_examples": _examples(te - ss), "sample_only_examples": _examples(ss - te),
        "train_and_test_examples": _examples(tr & te),
    }
    if len(test) != len(sample):
        r.add("error", "TEST_VS_SAMPLE_ROW_COUNT",
              f"test.csv has {len(test)} rows but sample_submission.csv has {len(sample)} rows")
    if te != ss:
        r.add("error", "TEST_VS_SAMPLE_FILENAMES",
              f"test.csv and sample_submission.csv filename sets differ: {len(te & ss)} shared, "
              f"{len(te - ss)} only in test.csv, {len(ss - te)} only in sample_submission.csv")
    if tr & te:
        r.add("warning", "TRAIN_TEST_NAME_OVERLAP",
              f"{len(tr & te)} filenames appear in BOTH train.csv and test.csv. Files live in separate folders, so this may "
              "be benign re-numbering, but verify the audio is not identical (run the audit with --hash-check).")
    if ss & tr:
        r.add("info", "SAMPLE_TRAIN_NAME_OVERLAP", f"{len(ss & tr)} sample_submission filenames also appear in train.csv")
    return r


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def audit_audio(report: AuditReport, paths: DataPaths, train: pd.DataFrame, test: pd.DataFrame,
                sample: pd.DataFrame, hash_check: bool = False, progress: bool = True) -> AuditReport:
    """Adds on-disk audio checks (missing/extra/invalid files, durations, sample rates, channels)."""
    from tqdm import tqdm

    rows = []
    audio_stats: dict = {}
    for split, df in (("train", train), ("test", test)):
        directory = paths.audio_dir(split)
        if not directory.is_dir():
            report.add("error", f"{split.upper()}_AUDIO_DIR_MISSING", f"audio directory not found: {directory}")
            continue
        on_disk = set(list_audio_files(directory))
        listed = set(df["filename"])
        missing, extra = listed - on_disk, on_disk - listed
        audio_stats[split] = {"files_on_disk": len(on_disk), "listed_in_csv": len(listed),
                              "missing_audio": len(missing), "extra_audio": len(extra),
                              "missing_examples": _examples(missing), "extra_examples": _examples(extra)}
        if missing:
            report.add("error", f"{split.upper()}_MISSING_AUDIO", f"{len(missing)} files listed in {split}.csv are not on disk")
        if extra:
            report.add("warning", f"{split.upper()}_EXTRA_AUDIO", f"{len(extra)} audio files on disk are not listed in {split}.csv")
        todo = sorted(on_disk)
        it = tqdm(todo, desc=f"probe {split}", disable=not progress)
        for fn in it:
            rec = probe_audio(directory / fn)
            rec["split"] = split
            rec["listed_in_csv"] = fn in listed
            rows.append(rec)

    # sample_submission filenames vs the test audio folder
    test_dir = paths.test_audio_dir
    if test_dir.is_dir():
        on_disk_test = set(list_audio_files(test_dir))
        ss = set(sample["filename"])
        audio_stats["sample_submission_vs_test_audio"] = {
            "present_in_test_dir": len(ss & on_disk_test), "absent_from_test_dir": len(ss - on_disk_test),
            "absent_examples": _examples(ss - on_disk_test)}
        if ss - on_disk_test:
            report.add("error", "SAMPLE_FILES_NOT_IN_TEST_DIR",
                       f"{len(ss - on_disk_test)} sample_submission filenames have no audio in the test folder")

    inv = pd.DataFrame(rows)
    if not inv.empty:
        report.inventory = inv
        for split, g in inv.groupby("split"):
            ok = g[g["ok"]]
            invalid = g[~g["ok"]]
            audio_stats.setdefault(split, {}).update({
                "invalid_files": int(len(invalid)), "invalid_examples": invalid["filename"].head(MAX_EXAMPLES).tolist(),
                "duration_sec": ok["duration"].describe().round(3).to_dict() if len(ok) else {},
                "sample_rates": {str(int(k)): int(v) for k, v in ok["sample_rate"].value_counts().items()},
                "channels": {str(int(k)): int(v) for k, v in ok["channels"].value_counts().items()},
            })
            if len(invalid):
                report.add("error", f"{split.upper()}_INVALID_AUDIO", f"{len(invalid)} unreadable/empty {split} audio files")
            if len(ok) and ((ok["duration"] < 30) | (ok["duration"] > 90)).any():
                n = int(((ok["duration"] < 30) | (ok["duration"] > 90)).sum())
                report.add("info", f"{split.upper()}_DURATION_OUTLIERS", f"{n} {split} files are outside 30-90 s (brief says ~45-60 s)")

    if hash_check:
        shared = sorted(set(train["filename"]) & set(test["filename"]))
        identical, compared = [], 0
        for fn in tqdm(shared, desc="hash overlap", disable=not progress):
            a, b = paths.train_audio_dir / fn, paths.test_audio_dir / fn
            if a.exists() and b.exists():
                compared += 1
                if _md5(a) == _md5(b):
                    identical.append(fn)
        audio_stats["train_test_hash_check"] = {"compared": compared, "identical_audio": len(identical),
                                                "identical_examples": _examples(identical)}
        if identical:
            report.add("error", "TRAIN_TEST_IDENTICAL_AUDIO", f"{len(identical)} same-named train/test files are byte-identical (leakage risk)")
        elif compared:
            report.add("info", "TRAIN_TEST_NAME_OVERLAP_BENIGN", f"all {compared} same-named train/test pairs have different audio content")
    report.stats["audio"] = audio_stats
    return report


def run_audit(paths: DataPaths, expected_train: Optional[int] = 769, expected_test: Optional[int] = 216,
              check_audio: bool = True, hash_check: bool = False, progress: bool = True) -> AuditReport:
    from src.data.loader import load_all_metadata

    meta = load_all_metadata(paths)
    report = audit_metadata(meta.train, meta.test, meta.sample, expected_train, expected_test)
    if check_audio:
        if not (paths.train_audio_dir.is_dir() or paths.test_audio_dir.is_dir()):
            report.add("warning", "AUDIO_NOT_CHECKED", "no audio directories found; only CSV metadata was audited")
        else:
            audit_audio(report, paths, meta.train, meta.test, meta.sample, hash_check=hash_check, progress=progress)
    return report


def assert_test_matches_sample(test: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Hard stop used before submission generation."""
    te, ss = set(test["filename"]), set(sample["filename"])
    if len(test) != len(sample) or te != ss:
        raise MetadataInconsistencyError(
            f"test.csv ({len(test)} rows) and sample_submission.csv ({len(sample)} rows) are inconsistent: "
            f"{len(te & ss)} shared filenames, {len(te - ss)} only in test.csv, {len(ss - te)} only in sample_submission.csv. "
            "Investigate which file defines the Kaggle test set before generating a submission."
        )
