"""Dense retrieval — owner: Kevin.

sentence-transformers embeddings (normalized), inner-product search with FAISS (numpy fallback when FAISS is not
installed). Chunk embeddings are cached as .npy in data/cache/dense, keyed by model + chunk ids and texts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from snakepp import cache
from snakepp.schemas import Chunk, Retrieved

Encoder = Callable[[list[str]], np.ndarray]
# BGE v1.5 models expect this prefix on queries (not on passages).
QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}
MAX_PASSAGE_CHARS = 2000


def sentence_transformer_encoder(model_name: str) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    return lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)


def _passage(chunk: Chunk) -> str:
    return f"{chunk.title}\n{chunk.text}"[:MAX_PASSAGE_CHARS]


class DenseIndex:
    def __init__(self, chunks: list[Chunk], model_name: str, encoder: Encoder | None = None) -> None:
        self.ids = [c.id for c in chunks]
        self.model_name = model_name
        self._encode = encoder or sentence_transformer_encoder(model_name)
        self.vectors = self._embed_cached([_passage(c) for c in chunks]).astype("float32")
        self._faiss = self._build_faiss(self.vectors)

    def _embed_cached(self, passages: list[str]) -> np.ndarray:
        path = Path(cache.CACHE_DIR) / "dense" / f"{cache.key(self.model_name, self.ids, passages)}.npy"
        if path.exists():
            return np.load(path)
        vectors = np.asarray(self._encode(passages), dtype="float32")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    @staticmethod
    def _build_faiss(vectors: np.ndarray):
        try:
            import faiss
        except ImportError:
            return None
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def search(self, query: str, k: int = 50) -> list[Retrieved]:
        q = np.asarray(self._encode([QUERY_PREFIX.get(self.model_name, "") + query]), dtype="float32")
        k = min(k, len(self.ids))
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            pairs = list(zip(idx[0].tolist(), scores[0].tolist(), strict=True))
        else:
            sims = self.vectors @ q[0]
            top = np.argsort(-sims, kind="stable")[:k]
            pairs = [(int(i), float(sims[i])) for i in top]
        return [
            Retrieved(chunk_id=self.ids[i], rank=r, score=float(s), stage="dense")
            for r, (i, s) in enumerate(pairs, start=1)
        ]
