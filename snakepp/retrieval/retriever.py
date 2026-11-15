"""Hybrid retriever: BM25 + dense -> RRF -> rerank -> version filter. Owner: Kevin.

Status: modes bm25, dense and hybrid (RRF), the cross-encoder reranker and the soft/hard version filter. The leaky
Stack Overflow corpus is still to come; a config that asks for it fails loudly instead of silently running without it.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from snakepp import paths
from snakepp.config import Config
from snakepp.ingest.chunks import load_chunks
from snakepp.retrieval import version_filter
from snakepp.retrieval.bm25 import BM25Index, Hits
from snakepp.retrieval.dense import EMBED_CACHE, DenseIndex, Encoder, load_encoder
from snakepp.retrieval.fusion import rrf
from snakepp.retrieval.rerank import RERANK_CACHE, Reranker, Scorer, load_cross_encoder
from snakepp.schemas import Chunk, Retrieved, Stage


def _check_supported(cfg: Config) -> None:
    rc = cfg.retrieval
    if rc.mode == "none":
        return
    todo = []
    if rc.corpus != "clean":
        todo.append("retrieval.corpus: leaky (needs Lucas's decontaminator)")
    if todo:
        raise NotImplementedError(f"config {cfg.name!r} needs retrieval features not built yet: " + "; ".join(todo))


def chunk_files(cfg: Config, root: Path = paths.CHUNKS) -> list[Path]:
    """Every chunk file of the enabled sources, in the agreed layout <source>/<library>/<version>.parquet."""
    return sorted(p for source in cfg.retrieval.sources for p in (root / source).glob("*/*.parquet"))


class Retriever:
    def __init__(
        self,
        chunks: Iterable[Chunk],
        cfg: Config,
        *,
        encode: Encoder | None = None,
        rerank_score: Scorer | None = None,
        cache_dir: Path | None = EMBED_CACHE,
        rerank_cache_dir: Path | None = RERANK_CACHE,
    ) -> None:
        _check_supported(cfg)
        self.cfg = cfg
        rc = cfg.retrieval
        kept = [c for c in chunks if c.source in rc.sources] if rc.mode != "none" else []
        self._chunks = {c.id: c for c in kept}
        self._bm25 = BM25Index(kept) if kept and rc.mode in ("bm25", "hybrid") else None
        self._dense = None
        if kept and rc.mode in ("dense", "hybrid"):
            encoder = encode or load_encoder(rc.dense_model)
            self._dense = DenseIndex(kept, encoder, rc.dense_model, cache_dir)
        self._reranker = None
        if kept and rc.rerank:
            scorer = rerank_score or load_cross_encoder(rc.rerank_model)
            self._reranker = Reranker(scorer, rc.rerank_model, rerank_cache_dir)

    @classmethod
    def from_config(cls, cfg: Config) -> Retriever:
        """Index every chunk parquet under data/chunks/ for the config's sources."""
        _check_supported(cfg)
        return cls(load_chunks(chunk_files(cfg)), cfg)

    def stage_lists(self, query: str, library: str | None, n: int) -> dict[Stage, Hits]:
        """Each stage's ranked candidates (for the trace command and debugging)."""
        lists: dict[Stage, Hits] = {}
        if self._bm25:
            lists["bm25"] = self._bm25.search(query, n, library)
        if self._dense:
            lists["dense"] = self._dense.search(query, n, library)
        if len(lists) == 2:
            lists["fused"] = rrf([[cid for cid, _ in hits] for hits in lists.values()], self.cfg.retrieval.rrf_k)
        if self._reranker and lists:
            candidates = list(lists.values())[-1][: self.cfg.retrieval.candidates]
            lists["reranked"] = self._reranker.rerank(query, [self._chunks[cid] for cid, _ in candidates])
        return lists

    def retrieve(self, query: str, library: str | None, target_versions: dict[str, str], k: int) -> list[Retrieved]:
        """Top-k hits, best first, rank from 1, `chunk` filled; the version filter runs on the final list."""
        if not self._chunks or k < 1:
            return []
        lists = self.stage_lists(query, library, max(k, self.cfg.retrieval.candidates))
        if not lists:
            return []
        stage: Stage = list(lists)[-1]  # the last stage that ran: reranked > fused > the single stage
        final = lists[stage]
        if self.cfg.version_filter.mode != "off":
            final, _ = version_filter.apply(final, self._chunks, target_versions, self.cfg.version_filter)
        return [
            Retrieved(chunk_id=cid, rank=i, score=score, stage=stage, chunk=self._chunks[cid])
            for i, (cid, score) in enumerate(final[:k], start=1)
        ]

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]


@lru_cache(maxsize=1)
def _store() -> dict[str, Chunk]:
    return {c.id: c for c in load_chunks(sorted(paths.CHUNKS.glob("*/*/*.parquet")))}


def get_chunk(chunk_id: str) -> Chunk:
    """Full chunk with attribution. Raises KeyError for an unknown id."""
    return _store()[chunk_id]
