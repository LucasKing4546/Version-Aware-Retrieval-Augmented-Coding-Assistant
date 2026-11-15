"""Hybrid retriever: BM25 + dense -> RRF -> rerank -> version filter. Owner: Kevin."""

from __future__ import annotations

from snakepp.config import Config
from snakepp.schemas import Chunk, Retrieved


class Retriever:
    @classmethod
    def from_config(cls, cfg: Config) -> Retriever:
        """Load the indexes named by cfg.retrieval (mode, sources, corpus variant)."""
        raise NotImplementedError("Kevin: BM25 + FAISS + RRF on the fixture (Week 8)")

    def retrieve(self, query: str, library: str | None, target_versions: dict[str, str], k: int) -> list[Retrieved]:
        """Top-k hits, best first, rank starting at 1, with `chunk` filled in."""
        raise NotImplementedError("Kevin")


def get_chunk(chunk_id: str) -> Chunk:
    """Full chunk with attribution. Raises KeyError for an unknown id."""
    raise NotImplementedError("Kevin")
