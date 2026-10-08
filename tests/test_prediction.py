import json

import numpy as np
import pandas as pd
import pytest

from src.asr.whisper_transcriber import resolve_compute_type, transcribe_manifest
from src.config import load_config
from src.evaluation.metrics import clip_predictions
from src.models.baseline import ConstantBaseline
from src.models.predict import load_model, predict_scores
from src.models.train import (LOG_COLUMNS, build_pipeline, fit_final, get_feature_importance, read_log,
                              row_to_estimator, run_experiments, save_final, select_best, update_log)
from src.submission import SubmissionError, submission_path, validate_submission, write_submission


# ------------------------------------------------------------------ clipping + prediction
class _Const:
    def __init__(self, v): self.v = np.asarray(v, float)
    def predict(self, X): return self.v


def test_prediction_clipping():
    feats = pd.DataFrame({"filename": ["a.wav", "b.wav", "c.wav"]})
    out = predict_scores(_Const([-0.7, 2.2, 6.3]), feats, 0, 5)
    assert out["label"].tolist() == [0.0, 2.2, 5.0]
    assert list(out.columns) == ["filename", "label"]
    with pytest.raises(ValueError):
        predict_scores(_Const([1, np.nan, 2]), feats)
    with pytest.raises(ValueError):
        predict_scores(_Const([1, 2]), pd.DataFrame({"filename": ["a", "a"]}))
    assert clip_predictions([9]).tolist() == [5.0]


# ------------------------------------------------------------------ submission validation
@pytest.fixture
def sample():
    return pd.DataFrame({"filename": ["a.wav", "b.wav", "c.wav"], "label": [-1, -1, -1]})


def test_valid_submission_is_reordered_to_sample_order(sample):
    sub = pd.DataFrame({"filename": ["c.wav", "a.wav", "b.wav"], "label": [3.0, 1.0, 2.0]})
    out = validate_submission(sub, sample)
    assert out["filename"].tolist() == ["a.wav", "b.wav", "c.wav"] and out["label"].tolist() == [1.0, 2.0, 3.0]
    assert list(out.columns) == list(sample.columns)


@pytest.mark.parametrize("mutate,msg", [
    (lambda s: s.rename(columns={"label": "score"}), "columns"),
    (lambda s: s.iloc[:2], "row count"),
    (lambda s: pd.concat([s, s.iloc[[0]]]), "duplicate"),
    (lambda s: s.assign(label=[1.0, np.nan, 2.0]), "missing"),
    (lambda s: s.assign(label=[1.0, 7.0, 2.0]), "outside"),
    (lambda s: s.assign(filename=["a.wav", "b.wav", "zzz.wav"]), "filename sets differ"),
])
def test_invalid_submissions_rejected(sample, mutate, msg):
    good = pd.DataFrame({"filename": sample["filename"], "label": [1.0, 2.0, 3.0]})
    with pytest.raises(SubmissionError, match=msg):
        validate_submission(mutate(good), sample)


def test_non_numeric_predictions_rejected(sample):
    bad = pd.DataFrame({"filename": sample["filename"], "label": ["x", "2", "3"]})
    with pytest.raises(SubmissionError):
        validate_submission(bad, sample)


def test_row_count_mismatch_with_extra_rows_rejected(sample):
    sub = pd.DataFrame({"filename": list(sample["filename"]) + ["d.wav"], "label": [1, 2, 3, 4.0]})
    with pytest.raises(SubmissionError, match="row count"):
        validate_submission(sub, sample)


def test_versioned_names_and_no_overwrite(tmp_path, sample):
    assert submission_path(tmp_path, 4, "fusion").name == "submission_v04_fusion.csv"
    assert submission_path(tmp_path, None, "final").name == "submission_final.csv"
    sub = validate_submission(pd.DataFrame({"filename": sample["filename"], "label": [1.0, 2.0, 3.0]}), sample)
    p = write_submission(sub, submission_path(tmp_path, 1, "baseline"))
    assert pd.read_csv(p).shape == (3, 2)
    with pytest.raises(SubmissionError, match="already exists"):
        write_submission(sub, p)
    write_submission(sub, p, overwrite=True)
    official = tmp_path / "sample_submission.csv"
    sample.to_csv(official, index=False)
    with pytest.raises(SubmissionError, match="official"):
        write_submission(sub, official, protected=official, overwrite=True)
    assert pd.read_csv(official)["label"].tolist() == [-1, -1, -1]   # untouched


# ------------------------------------------------------------------ ASR cache / resume
def test_asr_cache_resume_and_retry(tmp_path):
    (tmp_path / "audio").mkdir()
    for n in ["a.wav", "b.wav", "c.wav"]:
        (tmp_path / "audio" / n).write_bytes(b"x")
    calls, failed_once = [], set()

    def fake(path):
        calls.append(path.name)
        if path.name == "b.wav" and "b.wav" not in failed_once:
            failed_once.add("b.wav")
            raise RuntimeError("boom")
        return f"text of {path.stem}.", 50.0

    out = tmp_path / "t.csv"
    df = transcribe_manifest(["a.wav", "b.wav", "c.wav", "missing.wav"], tmp_path / "audio", out, fake, "m", progress=False, save_every=1)
    assert df.set_index("filename")["status"].to_dict() == {"a.wav": "ok", "b.wav": "error", "c.wav": "ok", "missing.wav": "error"}
    calls.clear()
    transcribe_manifest(["a.wav", "b.wav", "c.wav", "missing.wav"], tmp_path / "audio", out, fake, "m", progress=False)
    assert calls == []                         # everything cached; failures not retried by default
    df = transcribe_manifest(["a.wav", "b.wav", "c.wav"], tmp_path / "audio", out, fake, "m", retry_failed=True, progress=False)
    assert calls == ["b.wav"]                  # only the failed file re-ran
    assert df.set_index("filename").loc["b.wav", "status"] == "ok"
    calls.clear()
    transcribe_manifest(["a.wav"], tmp_path / "audio", out, fake, "m", force=True, progress=False)
    assert calls == ["a.wav"]
    assert set(pd.read_csv(out)["filename"]) == {"a.wav"}   # force rebuilds the cache
    assert list(pd.read_csv(out).columns) == ["filename", "transcript", "duration", "status", "error_message", "asr_model"]


def test_compute_type_resolution():
    assert resolve_compute_type("auto", "cuda") == "float16"
    assert resolve_compute_type("auto", "cpu") == "int8"
    assert resolve_compute_type("float32", "cpu") == "float32"


# ------------------------------------------------------------------ models + experiment log
def _train_table(n=150, seed=0):
    rng = np.random.default_rng(seed)
    y = np.clip(np.round(rng.normal(3.3, 1.2, n) * 2) / 2, 0, 5)
    d = {"filename": [f"audio_{i}.wav" for i in range(n)]}
    d["aud_x"] = y + rng.normal(0, 1, n)
    d["txt_n_words"] = 50 + 20 * y + rng.normal(0, 15, n)
    d["ling_r"] = rng.normal(size=n)
    for i in range(12):
        d[f"emb_{i:03d}"] = rng.normal(size=n) + (0.5 * y if i < 3 else 0)
    return pd.DataFrame(d), y


@pytest.fixture
def cfg():
    c = load_config()
    c["features"]["pca_components"] = 5
    return c


def test_constant_baseline():
    m = ConstantBaseline("median").fit(None, [1, 2, 10])
    assert m.predict(range(4)).tolist() == [2.0] * 4
    assert ConstantBaseline("mean").fit(None, [1, 2, 3]).predict([0])[0] == 2.0
    with pytest.raises(ValueError):
        ConstantBaseline("mode").fit(None, [1])


def test_experiment_runner_log_selection_and_final(tmp_path, cfg):
    df, y = _train_table()
    exps = [
        {"id": "EXP-00", "feature_set": [], "models": ["mean_baseline"], "notes": "n"},
        {"id": "EXP-04", "feature_set": ["audio", "text"], "models": ["extra_trees", "ridge"], "notes": ""},
        {"id": "EXP-06", "feature_set": ["audio", "text", "embeddings"], "models": ["extra_trees"], "notes": ""},
        {"id": "EXP-07", "type": "ensemble", "top_k": 2, "notes": ""},
    ]
    cfg["validation"]["folds"] = 3
    log_path, oof_path = tmp_path / "log.csv", tmp_path / "oof.csv"
    log = run_experiments(df, y, cfg, exps, log_path, oof_path, verbose=False)
    assert list(log.columns) == LOG_COLUMNS
    assert set(log["experiment_id"]) == {"EXP-00", "EXP-04", "EXP-06", "EXP-07"}
    base = log[log.model == "mean_baseline"].iloc[0]
    best = select_best(log)
    assert best["cv_rmse_mean"] < base["cv_rmse_mean"]            # signal is learnable
    assert best["model"] != "mean_baseline"
    et = log[(log.experiment_id == "EXP-04") & (log.model == "extra_trees")].iloc[0]
    assert et["training_rmse"] < et["cv_rmse_mean"]               # in-sample is optimistic
    oof = pd.read_csv(oof_path)
    assert {"filename", "y_true", "fold", "EXP-04__extra_trees"} <= set(oof.columns)
    assert oof["EXP-04__extra_trees"].between(0, 5).all()

    # user-entered Kaggle info survives a re-run
    log.loc[log.experiment_id == "EXP-04", "kaggle_score"] = 0.5
    log.to_csv(log_path, index=False)
    log2 = run_experiments(df, y, cfg, exps, log_path, oof_path, only=["EXP-04"], verbose=False)
    assert (log2[log2.experiment_id == "EXP-04"]["kaggle_score"] == 0.5).all()
    assert len(log2) == len(log)                                  # upsert, not duplicate

    model, tr_rmse = fit_final(best, df, y, cfg)
    path = save_final(model, best, tmp_path, "final_model", {"training_rmse": tr_rmse})
    loaded, meta = load_model(path)
    assert meta["model"] == best["model"] and "training_rmse" in meta
    pred = predict_scores(loaded, df.head(10))
    assert pred["label"].between(0, 5).all() and len(pred) == 10
    imp = get_feature_importance(model)
    if imp is not None:
        assert "aud_x" in set(imp["feature"]) or any(f.startswith("emb_pc") for f in imp["feature"])


def test_ensemble_estimator_roundtrip(cfg):
    df, y = _train_table()
    row = pd.Series({"experiment_id": "EXP-07", "model": "ensemble_top2", "feature_set": "mixed",
                     "hyperparameters": json.dumps({"members": [
                         {"experiment_id": "EXP-04", "model": "extra_trees", "feature_set": "audio+text"},
                         {"experiment_id": "EXP-06", "model": "ridge", "feature_set": "audio+text+embeddings"}]})})
    est = row_to_estimator(row, cfg).fit(df, y)
    assert est.predict(df).shape == (len(df),)


def test_pipeline_rejects_unknown_model(cfg):
    with pytest.raises(ValueError):
        build_pipeline("catboost", ["audio"], cfg)


def test_update_log_new_file(tmp_path):
    row = {c: np.nan for c in LOG_COLUMNS}
    row.update(experiment_id="EXP-X", model="m", feature_set="audio")
    out = update_log(tmp_path / "sub" / "l.csv", [row])
    assert len(out) == 1 and len(read_log(tmp_path / "sub" / "l.csv")) == 1
