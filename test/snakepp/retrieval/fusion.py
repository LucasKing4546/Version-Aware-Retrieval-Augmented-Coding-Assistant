"""Reciprocal rank fusion — owner: Kevin."""

from __future__ import annotations

from snakepp.schemas import Retrieved


def rrf(rankings: list[list[Retrieved]], k: int = 60, top_n: int | None = None) -> list[Retrieved]:
    """Merge several ranked lists: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for r in ranking:
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + r.rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_n is not None:
        ordered = ordered[:top_n]
    return [Retrieved(chunk_id=cid, rank=i + 1, score=s, stage="fused") for i, (cid, s) in enumerate(ordered)]
