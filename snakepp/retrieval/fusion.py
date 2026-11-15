"""Reciprocal rank fusion. Owner: Kevin.

RRF needs only ranks, so BM25 scores and cosine similarities never have to be put on one scale.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists (best first): score(id) = sum of 1 / (k + rank). Ties break by id."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
