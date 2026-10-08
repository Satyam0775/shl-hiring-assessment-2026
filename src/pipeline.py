"""End-to-end orchestration used by the CLI scripts and the notebooks.

Every stage is a cached, keyed table (one row per filename):
  transcripts  -> data/processed/transcripts/{split}_transcripts.csv
  audio feats  -> data/processed/audio_features/{split}_audio_features.csv
  text feats   -> data/processed/text_features/{split}_text_features.csv
  embeddings   -> data/processed/text_features/{split}_embeddings.csv
  merged       -> data/processed/merged_features/{split}_features.csv
Stages skip rows that already exist unless ``force=True``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DataPaths, build_data_paths, load_config, repo_dir, resolve_data_root
from src.data.loader import Metadata, get_manifest, load_all_metadata, resolve_audio_path
from src.features.fusion import merge_feature_tables, rows_missing_features
from src.utils import read_cache, upsert_rows, write_csv_atomic


@dataclass
class Context:
    cfg: dict
    paths: DataPaths

    def processed(self, sub: str) -> Path:
        p = repo_dir(self.cfg, "processed_dir") / sub
        p.mkdir(parents=True, exist_ok=True)
        return p

    def meta(self) -> Metadata:
        return load_all_metadata(self.paths)


def get_context(data_root: Optional[str] = None, config_path: Optional[str] = None) -> Context:
    cfg = load_config(config_path)
    root = resolve_data_root(data_root, cfg)
    return Context(cfg=cfg, paths=build_data_paths(root, cfg))


def _manifest_tag(manifest: str) -> str:
    return "" if manifest == "test" else f"__{manifest}"


def manifest_frame(ctx: Context, split: str, manifest: str = "test") -> pd.DataFrame:
    meta = ctx.meta()
    if split == "train":
        return meta.train
    return get_manifest(meta, manifest)


# ------------------------------------------------------------------ ASR
def run_transcription(ctx: Context, split: str, manifest: str = "test", force: bool = False,
                      retry_failed: bool = False, limit: Optional[int] = None) -> pd.DataFrame:
    from src.asr.whisper_transcriber import WhisperTranscriber, transcribe_manifest

    asr = ctx.cfg["asr"]
    df = manifest_frame(ctx, split, manifest)
    names = df["filename"].tolist()[:limit]
    tr = WhisperTranscriber(asr["model"], asr.get("fallback_model"), asr.get("language", "en"),
                            asr.get("compute_type", "auto"), int(asr.get("beam_size", 5)), bool(asr.get("vad_filter", False)))
    print(f"[asr] device={tr.device} compute_type={tr.compute_type} model={asr['model']}")
    out = ctx.processed("transcripts") / f"{split}_transcripts.csv"
    return transcribe_manifest(names, ctx.paths.audio_dir(split), out, tr.transcribe_file, asr["model"],
                               force=force, retry_failed=retry_failed, save_every=int(asr.get("save_every", 10)))


# ------------------------------------------------------------------ acoustic features
def run_audio_features(ctx: Context, split: str, manifest: str = "test", force: bool = False,
                       limit: Optional[int] = None) -> pd.DataFrame:
    from joblib import Parallel, delayed
    from tqdm import tqdm

    from src.audio.audio_features import extract_features_for_file

    out = ctx.processed("audio_features") / f"{split}_audio_features.csv"
    names = manifest_frame(ctx, split, manifest)["filename"].tolist()[:limit]
    cache = None if force else read_cache(out)
    done = set(cache.loc[cache["status"] == "ok", "filename"]) if cache is not None else set()
    todo = [n for n in names if n not in done]
    print(f"[audio] {len(names)} requested | {len(names) - len(todo)} cached | {len(todo)} to extract")
    if todo:
        cfg_a = ctx.cfg["audio"]
        gen = Parallel(n_jobs=int(ctx.cfg["features"].get("n_jobs", 1)), return_as="generator")(
            delayed(extract_features_for_file)(resolve_audio_path(ctx.paths, split, n), cfg_a) for n in todo)
        rows = list(tqdm(gen, total=len(todo), desc=f"audio feats {split}"))
        cache = upsert_rows(cache, pd.DataFrame(rows))
        write_csv_atomic(cache, out)
        bad = [r["filename"] for r in rows if r["status"] != "ok"]
        if bad:
            print(f"[audio] {len(bad)} files failed (see status/error_message), e.g. {bad[:5]}")
    return cache if cache is not None else pd.DataFrame()


# ------------------------------------------------------------------ text features
def run_text_features(ctx: Context, split: str, force: bool = False) -> pd.DataFrame:
    from src.text.linguistic_features import extract_linguistic_features
    from src.text.text_features import extract_text_features

    tpath = ctx.processed("transcripts") / f"{split}_transcripts.csv"
    tr = read_cache(tpath)
    if tr is None:
        raise FileNotFoundError(f"{tpath} not found - run scripts/transcribe.py first")
    out = ctx.processed("text_features") / f"{split}_text_features.csv"
    cache = None if force else read_cache(out)
    todo = tr if cache is None else tr[~tr["filename"].isin(cache["filename"])]
    rows = []
    for rec in todo.itertuples(index=False):
        text = rec.transcript if isinstance(rec.transcript, str) else ""
        ok = rec.status == "ok"
        row = {"filename": rec.filename}
        if ok:
            row.update(extract_text_features(text, None if pd.isna(rec.duration) else float(rec.duration)))
            row.update(extract_linguistic_features(text))
        rows.append(row)  # failed ASR -> NaN features (flagged later by rows_missing_features)
    if rows:
        cache = upsert_rows(cache, pd.DataFrame(rows))
        write_csv_atomic(cache, out)
    return cache


def run_embeddings(ctx: Context, split: str, force: bool = False) -> pd.DataFrame:
    from src.text.embeddings import embed_texts, embeddings_to_frame, load_embedder

    tpath = ctx.processed("transcripts") / f"{split}_transcripts.csv"
    tr = read_cache(tpath)
    if tr is None:
        raise FileNotFoundError(f"{tpath} not found - run scripts/transcribe.py first")
    tr = tr[tr["status"] == "ok"].copy()
    tr["transcript"] = tr["transcript"].fillna("")
    out = ctx.processed("text_features") / f"{split}_embeddings.csv"
    cache = None if force else read_cache(out)
    todo = tr if cache is None else tr[~tr["filename"].isin(cache["filename"])]
    if len(todo):
        f = ctx.cfg["features"]
        model = load_embedder(f["embedding_model"])
        arr = embed_texts(todo["transcript"].tolist(), model, int(f.get("embedding_batch_size", 32)))
        cache = upsert_rows(cache, embeddings_to_frame(todo["filename"], arr))
        write_csv_atomic(cache, out)
    return cache


# ------------------------------------------------------------------ fusion
def build_merged(ctx: Context, split: str, manifest: str = "test", force: bool = True, strict: bool = True) -> pd.DataFrame:
    """Merge cached feature tables onto the manifest. In strict mode, rows lacking features raise an error."""
    df = manifest_frame(ctx, split, manifest)
    f = ctx.cfg["features"]
    aud = read_cache(ctx.processed("audio_features") / f"{split}_audio_features.csv") if f.get("use_audio", True) else None
    txt = read_cache(ctx.processed("text_features") / f"{split}_text_features.csv") if f.get("use_text", True) else None
    emb = read_cache(ctx.processed("text_features") / f"{split}_embeddings.csv") if f.get("use_embeddings", True) else None
    base = df[["filename", "label"]].copy() if split == "train" else df[["filename"]].copy()
    merged = merge_feature_tables(base, aud, txt, emb)
    groups = [g for g, on, tbl in (("audio", f.get("use_audio", True), aud), ("text", f.get("use_text", True), txt),
                                   ("embeddings", f.get("use_embeddings", True), emb)) if on and tbl is not None]
    bad = rows_missing_features(merged, groups)
    if len(bad):
        msg = f"{len(bad)} {split} rows have no features for at least one enabled group, e.g. {bad['filename'].head(5).tolist()}"
        if strict:
            raise RuntimeError(msg + " (re-run the extraction stages / retry failed ASR, or pass allow-missing)")
        print("[warn]", msg)
    path = ctx.processed("merged_features") / f"{split}_features{_manifest_tag(manifest) if split == 'test' else ''}.csv"
    write_csv_atomic(merged, path)
    return merged


def load_train_matrix(ctx: Context) -> tuple[pd.DataFrame, np.ndarray]:
    df = build_merged(ctx, "train", strict=True)
    y = pd.to_numeric(df["label"], errors="raise").to_numpy(dtype=float)
    return df.drop(columns=["label"]), y
