import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.config import (ConfigError, build_data_paths, load_config, resolve_data_root, validate_config)
from src.data.loader import (SchemaError, check_schema, list_audio_files, load_all_metadata, read_metadata_csv,
                             resolve_audio_path)
from src.data.validator import (MetadataInconsistencyError, assert_test_matches_sample, audit_audio,
                                audit_metadata, run_audit)


def _write_dataset(root, train_names, test_names, sample_names, make_audio=False):
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"filename": train_names, "label": np.linspace(0, 5, len(train_names))}).to_csv(root / "train.csv", index=False)
    pd.DataFrame({"filename": test_names, "label": -1}).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"filename": sample_names, "label": -1}).to_csv(root / "sample_submission.csv", index=False)
    if make_audio:
        sig = (0.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype("float32")
        for split, names in (("train", train_names), ("test", test_names)):
            for n in names:
                sf.write(root / split / n, sig, 16000)


def test_config_loads_and_validates():
    cfg = load_config()
    assert cfg["validation"]["folds"] == 5
    assert cfg["prediction"]["min_score"] == 0 and cfg["prediction"]["max_score"] == 5
    assert cfg["asr"]["model"] == "small.en"


def test_config_validation_errors():
    cfg = load_config()
    bad = dict(cfg); bad.pop("asr")
    with pytest.raises(ConfigError):
        validate_config(bad)
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")


def test_data_root_precedence(monkeypatch, tmp_path):
    cfg = {"data_root": str(tmp_path / "cfg")}
    monkeypatch.delenv("SHL_DATA_ROOT", raising=False)
    assert resolve_data_root(None, cfg) == tmp_path / "cfg"
    monkeypatch.setenv("SHL_DATA_ROOT", str(tmp_path / "env"))
    assert resolve_data_root(None, cfg) == tmp_path / "env"
    assert resolve_data_root(str(tmp_path / "cli"), cfg) == tmp_path / "cli"
    monkeypatch.delenv("SHL_DATA_ROOT")
    with pytest.raises(ConfigError):
        resolve_data_root(None, {"data_root": None})


def test_path_resolution(tmp_path):
    paths = build_data_paths(tmp_path, load_config())
    assert paths.train_csv == tmp_path / "train.csv"
    assert paths.audio_dir("test") == tmp_path / "test"
    assert resolve_audio_path(paths, "train", "audio_1.wav") == tmp_path / "train" / "audio_1.wav"
    with pytest.raises(ValueError):
        resolve_audio_path(paths, "train", "../evil.wav")
    with pytest.raises(ValueError):
        paths.audio_dir("val")


def test_csv_loading_and_schema(tmp_path):
    _write_dataset(tmp_path, ["audio_1.wav", "audio_2.wav"], ["audio_0.wav"], ["audio_0.wav"])
    meta = load_all_metadata(build_data_paths(tmp_path, load_config()))
    assert len(meta.train) == 2 and len(meta.test) == 1
    pd.DataFrame({"name": ["a"], "score": [1]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(SchemaError):
        read_metadata_csv(tmp_path / "bad.csv", "bad")
    with pytest.raises(SchemaError):
        check_schema(pd.DataFrame({"filename": []}), "x")
    with pytest.raises(FileNotFoundError):
        read_metadata_csv(tmp_path / "nope.csv")


def test_missing_csvs_reports_expected_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="Expected dataset layout"):
        load_all_metadata(build_data_paths(tmp_path, load_config()))


def test_audit_detects_test_sample_mismatch():
    train = pd.DataFrame({"filename": ["audio_1.wav", "audio_2.wav"], "label": [1.0, 4.0]})
    test = pd.DataFrame({"filename": ["audio_1.wav", "audio_2.wav", "audio_3.wav"], "label": -1})
    sample = pd.DataFrame({"filename": ["audio_9.wav", "audio_1.wav"], "label": -1})
    r = audit_metadata(train, test, sample, expected_train=2, expected_test=3)
    codes = {f.code for f in r.errors}
    assert {"TEST_VS_SAMPLE_ROW_COUNT", "TEST_VS_SAMPLE_FILENAMES"} <= codes
    assert r.stats["overlap"]["test_and_sample"] == 1
    assert r.stats["overlap"]["train_and_test"] == 2
    assert any(f.code == "TRAIN_TEST_NAME_OVERLAP" for f in r.warnings)
    with pytest.raises(MetadataInconsistencyError):
        assert_test_matches_sample(test, sample)


def test_audit_clean_and_label_range():
    names = ["audio_1.wav", "audio_2.wav"]
    train = pd.DataFrame({"filename": names, "label": [1.0, 9.0]})
    test = pd.DataFrame({"filename": ["audio_5.wav"], "label": -1})
    r = audit_metadata(train, test, test.copy(), expected_train=2, expected_test=1)
    assert [f.code for f in r.errors] == ["TRAIN_LABEL_RANGE"]
    ok = audit_metadata(train.assign(label=[1.0, 2.0]), test, test.copy(), expected_train=2, expected_test=1)
    assert not ok.has_errors
    assert_test_matches_sample(test, test.copy())


def test_audit_flags_duplicates_and_nans():
    train = pd.DataFrame({"filename": ["audio_1.wav", "audio_1.wav"], "label": [1.0, np.nan]})
    test = pd.DataFrame({"filename": ["audio_5.wav"], "label": -1})
    r = audit_metadata(train, test, test.copy(), None, None)
    codes = {f.code for f in r.errors}
    assert "TRAIN_DUPLICATES" in codes and "TRAIN_MISSING_VALUES" in codes


def test_audio_audit_with_real_wavs_and_problems(tmp_path):
    _write_dataset(tmp_path, ["audio_1.wav", "audio_2.wav"], ["audio_1.wav", "audio_7.wav"],
                   ["audio_1.wav", "audio_7.wav"], make_audio=True)
    (tmp_path / "train" / "audio_2.wav").unlink()                 # missing
    (tmp_path / "train" / "audio_99.wav").write_bytes(b"not a wav")  # extra + invalid
    paths = build_data_paths(tmp_path, load_config())
    report = run_audit(paths, expected_train=2, expected_test=2, progress=False, hash_check=True)
    codes = {f.code for f in report.findings}
    assert "TRAIN_MISSING_AUDIO" in codes and "TRAIN_EXTRA_AUDIO" in codes and "TRAIN_INVALID_AUDIO" in codes
    assert report.stats["audio"]["train"]["sample_rates"] == {"16000": 1}
    assert "TRAIN_TEST_IDENTICAL_AUDIO" in codes     # audio_1.wav identical in both folders
    assert list_audio_files(tmp_path / "test") == ["audio_1.wav", "audio_7.wav"]


def test_audio_dir_missing_is_reported(tmp_path):
    _write_dataset(tmp_path, ["audio_1.wav"], ["audio_2.wav"], ["audio_2.wav"])
    import shutil; shutil.rmtree(tmp_path / "train")
    paths = build_data_paths(tmp_path, load_config())
    r = audit_metadata(*[pd.read_csv(p) for p in (paths.train_csv, paths.test_csv, paths.sample_submission_csv)], 1, 1)
    audit_audio(r, paths, *[pd.read_csv(p) for p in (paths.train_csv, paths.test_csv, paths.sample_submission_csv)], progress=False)
    assert any(f.code == "TRAIN_AUDIO_DIR_MISSING" for f in r.errors)
