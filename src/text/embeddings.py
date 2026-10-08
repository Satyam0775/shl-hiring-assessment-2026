"""Sentence embeddings (columns ``emb_000`` ...). Local open-source model, no API key."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def embed_texts(texts: Sequence[str], model, batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
    """L2-normalised embeddings, shape (n, dim). ``model`` only needs an ``encode`` method."""
    clean = [t if isinstance(t, str) else "" for t in texts]
    arr = model.encode(clean, batch_size=batch_size, show_progress_bar=show_progress,
                       normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(arr, dtype=np.float32)


def embeddings_to_frame(filenames: Sequence[str], arr: np.ndarray) -> pd.DataFrame:
    cols = [f"emb_{i:03d}" for i in range(arr.shape[1])]
    df = pd.DataFrame(arr, columns=cols)
    df.insert(0, "filename", list(filenames))
    return df
