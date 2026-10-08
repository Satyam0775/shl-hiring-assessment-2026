"""Surface text features from an ASR transcript (columns prefixed ``txt_``)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import numpy as np

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
FILLERS = frozenset({"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "mm", "mmm", "uhm"})


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]


def extract_text_features(transcript: str, duration: Optional[float] = None) -> dict[str, float]:
    """Interpretable proxies. Empty transcripts yield zeros (a meaningful signal: no recognisable speech)."""
    transcript = transcript if isinstance(transcript, str) else ""
    words = tokenize(transcript)
    sents = split_sentences(transcript)
    n_words, n_sents = len(words), len(sents)
    sent_lens = np.array([len(tokenize(s)) for s in sents], dtype=float) if sents else np.array([], dtype=float)
    word_lens = np.array([len(w.replace("'", "")) for w in words], dtype=float)
    counts = Counter(words)
    vocab = len(counts)
    repeated = sum(1 for a, b in zip(words, words[1:]) if a == b)
    fillers = sum(c for w, c in counts.items() if w in FILLERS)

    f: dict[str, float] = {
        "txt_n_words": float(n_words),
        "txt_n_sentences": float(n_sents),
        "txt_avg_sent_len": float(sent_lens.mean()) if n_sents else 0.0,
        "txt_median_sent_len": float(np.median(sent_lens)) if n_sents else 0.0,
        "txt_max_sent_len": float(sent_lens.max()) if n_sents else 0.0,
        "txt_std_sent_len": float(sent_lens.std()) if n_sents else 0.0,
        "txt_vocab_size": float(vocab),
        "txt_unique_ratio": vocab / n_words if n_words else 0.0,
        "txt_root_ttr": vocab / np.sqrt(n_words) if n_words else 0.0,
        "txt_repeated_word_ratio": repeated / n_words if n_words else 0.0,
        "txt_filler_count": float(fillers),
        "txt_filler_ratio": fillers / n_words if n_words else 0.0,
        "txt_n_commas": float(transcript.count(",")),
        "txt_commas_per_sent": transcript.count(",") / n_sents if n_sents else 0.0,
        "txt_n_periods": float(transcript.count(".")),
        "txt_n_questions": float(transcript.count("?")),
        "txt_n_exclamations": float(transcript.count("!")),
        "txt_avg_word_len": float(word_lens.mean()) if n_words else 0.0,
        "txt_word_len_var": float(word_lens.var()) if n_words else 0.0,
        "txt_n_chars": float(len(transcript)),
    }
    f["txt_words_per_sec"] = (n_words / duration) if (duration and duration > 0) else np.nan
    return f
