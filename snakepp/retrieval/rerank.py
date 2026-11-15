"""Cross-encoder reranker over the fused candidates. Owner: Kevin.

A cross-encoder reads query and chunk together, so it fixes "right area, wrong chunk" results that BM25 and
embeddings both rank highly. It is the slowest retrieval step, so every (model, query, text) score is cached
in data/cache/rerank/<model>.json.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

from snakepp import paths
from snakepp.retrieval.bm25 import Hits
from snakepp.schemas import Chunk

Scorer = Callable[[str, list[str]], Sequence[float]]  # (query, passages) -> relevance, higher is better
RERANK_CACHE = paths.CACHE / "rerank"


@lru_cache(maxsize=2)
def load_cross_encoder(model_name: str) -> Scorer:
    """The cross-encoder model, loaded once per process."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)

    def score(query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in model.predict([(query, p) for p in passages], batch_size=16, show_progress_bar=False)]

    return score


class Reranker:
    def __init__(self, score: Scorer, model_name: str, cache_dir: Path | None = RERANK_CACHE) -> None:
        self._score = score
        self._model = model_name
        self._path = Path(cache_dir) / f"{re.sub(r'[^A-Za-z0-9._-]', '_', model_name)}.json" if cache_dir else None
        self._cache: dict[str, float] | None = None

    def _load(self) -> dict[str, float]:
        if self._cache is None:
            exists = self._path is not None and self._path.exists()
            self._cache = json.loads(self._path.read_text(encoding="utf-8")) if exists else {}
        return self._cache

    def _key(self, query: str, text: str) -> str:
        return hashlib.sha1(f"{self._model}\x1f{query}\x1f{text}".encode()).hexdigest()

    def rerank(self, query: str, candidates: Sequence[Chunk]) -> Hits:
        """All candidates, re-sorted by cross-encoder score (ties keep the incoming order)."""
        cache = self._load()
        keys = [self._key(query, c.text) for c in candidates]
        todo = {k: c.text for k, c in zip(keys, candidates, strict=True) if k not in cache}
        if todo:
            cache.update(zip(todo, self._score(query, list(todo.values())), strict=True))
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(cache), encoding="utf-8")
        scored = [(c.id, cache[k]) for c, k in zip(candidates, keys, strict=True)]
        return sorted(scored, key=lambda hit: -hit[1])
