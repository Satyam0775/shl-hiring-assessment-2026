"""Configuration loading and data-root resolution (Windows-safe, pathlib only)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

ENV_DATA_ROOT = "SHL_DATA_ROOT"
REQUIRED_KEYS = (
    "train_csv", "test_csv", "sample_submission_csv", "train_audio_dir",
    "test_audio_dir", "random_seed", "paths", "audio", "asr", "features",
    "validation", "prediction",
)


def project_root() -> Path:
    """Repository root (the folder that contains ``src/``)."""
    return Path(__file__).resolve().parents[1]


DEFAULT_CONFIG_PATH = project_root() / "configs" / "config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration is invalid or the data root is unusable."""


def validate_config(cfg: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ConfigError(f"Config is missing required keys: {missing}")
    p = cfg["prediction"]
    if not p["min_score"] < p["max_score"]:
        raise ConfigError("prediction.min_score must be < prediction.max_score")
    if int(cfg["validation"]["folds"]) < 2:
        raise ConfigError("validation.folds must be >= 2")


def load_config(path: Optional[os.PathLike | str] = None) -> dict[str, Any]:
    """Load and validate the YAML config (UTF-8)."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    validate_config(cfg)
    return cfg


@dataclass(frozen=True)
class DataPaths:
    """All dataset locations derived from a single data root."""
    root: Path
    train_csv: Path
    test_csv: Path
    sample_submission_csv: Path
    train_audio_dir: Path
    test_audio_dir: Path

    def audio_dir(self, split: str) -> Path:
        if split == "train":
            return self.train_audio_dir
        if split == "test":
            return self.test_audio_dir
        raise ValueError(f"Unknown split '{split}' (expected 'train' or 'test')")

    def missing_csvs(self) -> list[Path]:
        return [p for p in (self.train_csv, self.test_csv, self.sample_submission_csv)
                if not p.exists()]


def resolve_data_root(cli_value: Optional[str] = None, cfg: Optional[dict] = None) -> Path:
    """Resolve the dataset root: CLI flag > SHL_DATA_ROOT env var > config key."""
    candidates = [
        ("--data-root", cli_value),
        (ENV_DATA_ROOT, os.environ.get(ENV_DATA_ROOT)),
        ("config.data_root", (cfg or {}).get("data_root")),
    ]
    for _, value in candidates:
        if value:
            root = Path(os.path.expandvars(os.path.expanduser(str(value))))
            return root
    raise ConfigError(
        "No data root configured. Pass --data-root, set the SHL_DATA_ROOT "
        "environment variable, or set data_root in configs/config.yaml.\n"
        'PowerShell example: $env:SHL_DATA_ROOT = "C:/Users/USERNAME/path/Dataset_Final"'
    )


def build_data_paths(root: os.PathLike | str, cfg: dict) -> DataPaths:
    root = Path(root)
    return DataPaths(
        root=root,
        train_csv=root / cfg["train_csv"],
        test_csv=root / cfg["test_csv"],
        sample_submission_csv=root / cfg["sample_submission_csv"],
        train_audio_dir=root / cfg["train_audio_dir"],
        test_audio_dir=root / cfg["test_audio_dir"],
    )


def repo_dir(cfg: dict, key: str) -> Path:
    """Resolve a ``paths.*`` entry relative to the repo root (absolute paths kept)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else project_root() / p
