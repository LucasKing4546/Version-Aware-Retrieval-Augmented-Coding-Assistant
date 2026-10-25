"""Cross-encoder reranking — owner: Kevin.

Scores (query, chunk) pairs with a CrossEncoder, maps logits to (0, 1) with a sigmoid so the soft version
penalty can multiply them, and caches the scores in data/cache/rerank by model + query + chunk ids.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from snakepp import cache
from snakepp.schemas import Chunk, Retrieved

Scorer = Callable[[list[tuple[str, str]]], list[float]]
MAX_PASSAGE_CHARS = 2000


def cross_encoder_scorer(model_name: str) -> Scorer:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device="cpu")
    return lambda pairs: [float(s) for s in model.predict(pairs, batch_size=32)]


class Reranker:
    def __init__(self, model_name: str, scorer: Scorer | None = None) -> None:
        self.model_name = model_name
        self._scorer = scorer  # loaded on first use

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._scorer is None:
            self._scorer = cross_encoder_scorer(self.model_name)
        return self._scorer(pairs)

    def rerank(self, query: str, candidates: list[Retrieved], chunks: dict[str, Chunk]) -> list[Retrieved]:
        if not candidates:
            return []
        ids = [r.chunk_id for r in candidates]
        pairs = [(query, f"{chunks[i].title}\n{chunks[i].text}"[:MAX_PASSAGE_CHARS]) for i in ids]
        logits = cache.cached_json("rerank", cache.key(self.model_name, query, ids), lambda: self._score(pairs))
        scored = sorted(zip(ids, logits, strict=True), key=lambda x: -x[1])
        return [
            Retrieved(chunk_id=cid, rank=r, score=1 / (1 + math.exp(-s)), stage="reranked")
            for r, (cid, s) in enumerate(scored, start=1)
        ]
