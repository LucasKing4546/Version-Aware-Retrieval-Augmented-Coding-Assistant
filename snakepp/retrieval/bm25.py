"""BM25 over chunk title + text with the identifier-aware tokenizer. Owner: Kevin."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from snakepp.retrieval.tokenizer import tokenize
from snakepp.schemas import Chunk

Hits = list[tuple[str, float]]  # (chunk_id, score), best first


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        self.ids = [c.id for c in chunks]
        self._library = np.array([c.library for c in chunks])
        self._bm25 = BM25Okapi([tokenize(f"{c.title}\n{c.text}") for c in chunks])

    def search(self, query: str, n: int, library: str | None = None) -> Hits:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        if library is not None:
            scores = np.where(self._library == library, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:n]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]
