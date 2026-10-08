"""Shared CLI helpers (puts the repo root on sys.path so ``import src`` works from any cwd)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def base_parser(description: str, need_data_root: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to config YAML (default configs/config.yaml)")
    if need_data_root:
        p.add_argument("--data-root", default=None,
                       help="dataset folder containing train/, test/, train.csv, test.csv, sample_submission.csv "
                            "(falls back to $SHL_DATA_ROOT, then config.data_root)")
    return p


def add_split_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--split", choices=["train", "test", "both"], default="both")
    p.add_argument("--manifest", choices=["test", "sample_submission"], default="test",
                   help="which CSV lists the test files to process (default test.csv). Audio is always read from the test folder.")
    p.add_argument("--force", action="store_true", help="recompute even if cached")
    p.add_argument("--limit", type=int, default=None, help="process only the first N files (smoke test)")


def splits(arg: str) -> list[str]:
    return ["train", "test"] if arg == "both" else [arg]


def get_ctx(args):
    from src.pipeline import get_context

    return get_context(args.data_root, args.config)
