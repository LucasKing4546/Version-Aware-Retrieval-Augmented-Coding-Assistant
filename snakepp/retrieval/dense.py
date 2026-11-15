"""Dense retrieval: sentence embeddings in an exact in-process FAISS index. Owner: Kevin.

Embeddings are cached on disk per model, keyed by a hash of the chunk text, so re-indexing only embeds new or
changed chunks, and the docs and Stack Overflow retrievers of one config share the work.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from snakepp import paths
from snakepp.retrieval.bm25 import Hits
from snakepp.schemas import Chunk

Encoder = Callable[[list[str]], np.ndarray]  # texts -> L2-normalised float32 rows
EMBED_CACHE = paths.CACHE / "embeddings"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=2)
def load_encoder(model_name: str) -> Encoder:
    """The sentence-transformers model, loaded once per process and shared by every index."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> np.ndarray:
        vecs = model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype="float32")

    return encode


def query_prefix(model_name: str) -> str:
    return BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""


def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_cached(chunks: Sequence[Chunk], encode: Encoder, model_name: str, cache_dir: Path | None) -> np.ndarray:
    """Embeddings for `chunks` in order, encoding only texts missing from the model's cache file."""
    path = Path(cache_dir) / f"{re.sub(r'[^A-Za-z0-9._-]', '_', model_name)}.npz" if cache_dir else None
    stored: dict[str, np.ndarray] = {}
    if path and path.exists():
        with np.load(path) as f:
            stored = dict(zip(f["keys"].tolist(), f["vectors"], strict=True))
    keys = [_text_key(c.text) for c in chunks]
    missing = {k: c.text for k, c in zip(keys, chunks, strict=True) if k not in stored}
    if missing:
        stored.update(zip(missing, encode(list(missing.values())), strict=True))
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, keys=np.array(list(stored)), vectors=np.stack(list(stored.values())))
    return np.stack([stored[k] for k in keys]).astype("float32")


class DenseIndex:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        encode: Encoder,
        model_name: str,
        cache_dir: Path | None = EMBED_CACHE,
    ) -> None:
        if not chunks:
            raise ValueError("DenseIndex needs at least one chunk")
        self.ids = [c.id for c in chunks]
        self._library = np.array([c.library for c in chunks])
        self._encode = encode
        self._prefix = query_prefix(model_name)
        vectors = embed_cached(chunks, encode, model_name, cache_dir)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def search(self, query: str, n: int, library: str | None = None) -> Hits:
        params = None
        if library is not None:
            allowed = np.flatnonzero(self._library == library).astype("int64")
            if allowed.size == 0:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))
        q = self._encode([self._prefix + query])
        scores, idx = self._index.search(q, min(n, self._index.ntotal), params=params)
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0], strict=True) if i >= 0]
