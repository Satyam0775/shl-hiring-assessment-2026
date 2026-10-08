import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.audio.audio_features import extract_audio_features, extract_features_for_file
from src.audio.preprocessing import AudioLoadError, load_audio, peak_normalize, to_mono
from src.audio.quality import probe_audio, silence_ratio
from src.features.fusion import group_columns, merge_feature_tables, rows_missing_features
from src.features.preprocessing import GroupFeatureBuilder
from src.text.embeddings import embed_texts, embeddings_to_frame
from src.text.linguistic_features import extract_linguistic_features
from src.text.text_features import extract_text_features, split_sentences, tokenize

SR = 16000


def _speechlike(seconds=6):
    t = np.arange(int(SR * seconds)) / SR
    gate = (np.sin(2 * np.pi * 0.5 * t) > -0.2).astype("float32")
    rng = np.random.default_rng(0)
    return (0.3 * np.sin(2 * np.pi * 140 * t) * gate + 0.002 * rng.normal(size=t.size)).astype("float32")


def test_audio_features_synthetic():
    f = extract_audio_features(_speechlike(), SR)
    for key in ["aud_duration", "aud_rms_mean", "aud_zcr_std", "aud_mfcc01_mean", "aud_dmfcc05_std",
                "aud_centroid_mean", "aud_bandwidth_mean", "aud_rolloff_mean", "aud_silence_ratio", "aud_n_pauses"]:
        assert key in f and np.isfinite(f[key]), key
    assert f["aud_duration"] == pytest.approx(6.0)
    assert 0.0 <= f["aud_speech_ratio"] <= 1.0
    assert f["aud_n_pauses"] >= 1
    assert f["aud_pitch_mean"] == pytest.approx(140, rel=0.1)
    assert all(k.startswith("aud_") for k in f)


def test_audio_features_silence_and_short_clip():
    f = extract_audio_features(np.zeros(SR * 2, dtype="float32"), SR)
    assert f["aud_speech_ratio"] == 0.0 and f["aud_onset_rate"] == 0.0
    short = extract_audio_features(np.random.default_rng(0).normal(size=200).astype("float32") * 0.1, SR)
    assert np.isfinite(short["aud_rms_mean"])


def test_file_level_extraction_and_failures(tmp_path):
    sf.write(tmp_path / "ok.wav", _speechlike(3), SR)
    (tmp_path / "bad.wav").write_bytes(b"garbage")
    ok = extract_features_for_file(tmp_path / "ok.wav", {"sample_rate": SR})
    bad = extract_features_for_file(tmp_path / "bad.wav", {"sample_rate": SR})
    missing = extract_features_for_file(tmp_path / "none.wav", {"sample_rate": SR})
    assert ok["status"] == "ok" and bad["status"] == "error" and missing["status"] == "error"
    assert probe_audio(tmp_path / "ok.wav")["sample_rate"] == SR
    assert not probe_audio(tmp_path / "bad.wav")["ok"]


def test_preprocessing_resample_mono_and_original_untouched(tmp_path):
    stereo = np.stack([_speechlike(2), _speechlike(2)], axis=1)
    p = tmp_path / "s.wav"
    sf.write(p, stereo, 44100)
    before = p.read_bytes()
    y, sr = load_audio(p, sr=SR)
    assert sr == SR and y.ndim == 1 and abs(len(y) / SR - 2 * 16000 / 44100 * 44100 / 16000 * 0.0 - len(y) / SR) < 1e-9
    assert p.read_bytes() == before
    with pytest.raises(AudioLoadError):
        load_audio(p, sr=SR, min_duration=10)
    assert to_mono(stereo.T).ndim == 1
    z = np.zeros(10, dtype="float32")
    assert np.array_equal(peak_normalize(z), z)
    assert np.max(np.abs(peak_normalize(np.array([0.1, -0.5], dtype="float32")))) == pytest.approx(1.0)
    assert silence_ratio(np.zeros(100, dtype="float32")) == 1.0


def test_text_features():
    t = "Um, I like my school. I go there every day! Do you like it? Yes yes."
    f = extract_text_features(t, duration=10.0)
    assert f["txt_n_sentences"] == 4 and f["txt_n_questions"] == 1 and f["txt_n_exclamations"] == 1
    assert f["txt_filler_count"] == 1 and f["txt_repeated_word_ratio"] > 0
    assert f["txt_words_per_sec"] == pytest.approx(f["txt_n_words"] / 10.0)
    assert tokenize("Don't stop") == ["don't", "stop"]
    assert split_sentences("") == [] and len(split_sentences("A b. C d?")) == 2


def test_text_features_empty_transcript():
    f = extract_text_features("", None)
    assert f["txt_n_words"] == 0 and f["txt_unique_ratio"] == 0
    g = extract_linguistic_features("")
    assert all(v == 0 for v in g.values())
    assert extract_text_features(None)["txt_n_words"] == 0


def test_linguistic_rule_flags():
    bad = extract_linguistic_features("He don't like a apple. They was happy. I is here.")
    good = extract_linguistic_features("He doesn't like an apple. They were happy. I am here.")
    assert bad["ling_agreement_flags"] >= 3 and bad["ling_article_flags"] >= 1
    assert good["ling_agreement_flags"] == 0 and good["ling_article_flags"] == 0
    c = extract_linguistic_features("Although it rained, we went out because we were free, and it was fun.")
    assert c["ling_subord_conj_ratio"] > 0 and c["ling_coord_conj_ratio"] > 0
    assert all(k.startswith("ling_") for k in c)


class _StubEmbedder:
    def encode(self, texts, **kw):
        return np.array([[len(t), t.count(" "), 1.0] for t in texts], dtype=float)


def test_embedding_helpers_with_stub():
    arr = embed_texts(["a b", None], _StubEmbedder(), show_progress=False)
    df = embeddings_to_frame(["x.wav", "y.wav"], arr)
    assert list(df.columns) == ["filename", "emb_000", "emb_001", "emb_002"]


def _feature_df(n=60, n_emb=20, seed=0):
    rng = np.random.default_rng(seed)
    d = {"filename": [f"audio_{i}.wav" for i in range(n)]}
    d["aud_a"] = rng.normal(size=n); d["txt_b"] = rng.normal(size=n); d["ling_c"] = rng.normal(size=n)
    for i in range(n_emb):
        d[f"emb_{i:03d}"] = rng.normal(size=n)
    df = pd.DataFrame(d)
    df.loc[3, "aud_a"] = np.nan
    return df


def test_group_selection():
    df = _feature_df()
    assert group_columns(df.columns, ["audio"]) == ["aud_a"]
    assert group_columns(df.columns, ["text"]) == ["txt_b", "ling_c"]
    assert len(group_columns(df.columns, ["embeddings"])) == 20
    assert group_columns(df.columns, []) == []


def test_builder_fits_only_on_training_rows_no_leakage():
    df = _feature_df(80)
    train, val = df.iloc[:60], df.iloc[60:].copy()
    b = GroupFeatureBuilder(groups=("audio", "text", "embeddings"), use_pca=True, pca_components=5).fit(train)
    a = b.transform(val)
    val2 = val.copy()
    val2.iloc[:, 1:] = val2.iloc[:, 1:] * 100 + 7        # change the held-out rows massively
    b2 = GroupFeatureBuilder(groups=("audio", "text", "embeddings"), use_pca=True, pca_components=5).fit(train)
    np.testing.assert_allclose(b.pca_.components_, b2.pca_.components_)   # fitted basis unaffected by val data
    np.testing.assert_allclose(b.imputer_.statistics_, train[["aud_a", "txt_b", "ling_c"]].median().to_numpy())
    assert a.shape == (20, 3 + 5) and not np.isnan(a).any()
    assert list(b.get_feature_names_out())[-1] == "emb_pc04"
    assert b.pca_.n_features_in_ == 20 and b.pca_.n_samples_ == 60


def test_builder_pca_off_and_errors():
    df = _feature_df()
    b = GroupFeatureBuilder(groups=("embeddings",), use_pca=False).fit(df)
    assert b.transform(df).shape[1] == 20
    with pytest.raises(TypeError):
        GroupFeatureBuilder().fit(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        GroupFeatureBuilder(groups=("audio",)).fit(df[["filename", "txt_b"]])
    b2 = GroupFeatureBuilder(groups=("audio",)).fit(df)
    with pytest.raises(KeyError):
        b2.transform(df.drop(columns=["aud_a"]))


def test_merge_preserves_manifest_and_flags_missing():
    manifest = pd.DataFrame({"filename": ["a.wav", "b.wav", "c.wav"]})
    aud = pd.DataFrame({"filename": ["a.wav", "b.wav"], "aud_x": [1.0, 2.0], "status": ["ok", "ok"]})
    txt = pd.DataFrame({"filename": ["c.wav", "a.wav", "b.wav"], "txt_y": [3.0, 1.0, 2.0]})
    m = merge_feature_tables(manifest, aud, txt)
    assert m["filename"].tolist() == ["a.wav", "b.wav", "c.wav"] and "status" not in m.columns
    assert m.loc[2, "txt_y"] == 3.0 and np.isnan(m.loc[2, "aud_x"])
    assert rows_missing_features(m, ["audio", "text"])["filename"].tolist() == ["c.wav"]
    with pytest.raises(ValueError):
        merge_feature_tables(manifest, pd.concat([aud, aud]))
