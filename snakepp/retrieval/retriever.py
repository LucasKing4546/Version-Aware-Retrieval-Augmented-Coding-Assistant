"""Hybrid retriever: BM25 + dense -> RRF -> rerank -> version filter. Owner: Kevin.

Week 8 status: modes bm25, dense and hybrid (RRF) work. The cross-encoder reranker, the version filter and the
leaky Stack Overflow corpus arrive in Week 9; a config that asks for them fails loudly instead of silently
running without them.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from snakepp import paths
from snakepp.config import Config
from snakepp.ingest.chunks import load_chunks
from snakepp.retrieval.bm25 import BM25Index, Hits
from snakepp.retrieval.dense import EMBED_CACHE, DenseIndex, Encoder, load_encoder
from snakepp.retrieval.fusion import rrf
from snakepp.schemas import Chunk, Retrieved, Stage


def _check_supported(cfg: Config) -> None:
    rc = cfg.retrieval
    if rc.mode == "none":
        return
    todo = []
    if rc.rerank:
        todo.append("retrieval.rerank: true (set it to false until the reranker lands)")
    if cfg.version_filter.mode != "off":
        todo.append(f"version_filter.mode: {cfg.version_filter.mode}")
    if rc.corpus != "clean":
        todo.append("retrieval.corpus: leaky (needs Lucas's decontaminator)")
    if todo:
        raise NotImplementedError(f"config {cfg.name!r} needs Week 9 retrieval features: " + "; ".join(todo))


def chunk_files(cfg: Config, root: Path = paths.CHUNKS) -> list[Path]:
    """Every parquet file of the sources the config enables."""
    return sorted(p for source in cfg.retrieval.sources for p in (root / source).rglob("*.parquet"))


class Retriever:
    def __init__(
        self,
        chunks: Iterable[Chunk],
        cfg: Config,
        *,
        encode: Encoder | None = None,
        cache_dir: Path | None = EMBED_CACHE,
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
        return lists

    def retrieve(self, query: str, library: str | None, target_versions: dict[str, str], k: int) -> list[Retrieved]:
        """Top-k hits, best first, rank from 1, `chunk` filled. target_versions is used once the filter lands."""
        if not self._chunks or k < 1:
            return []
        lists = self.stage_lists(query, library, max(k, self.cfg.retrieval.candidates))
        stage: Stage = list(lists)[-1]  # fused when both ran, otherwise the single stage
        return [
            Retrieved(chunk_id=cid, rank=i, score=score, stage=stage, chunk=self._chunks[cid])
            for i, (cid, score) in enumerate(lists[stage][:k], start=1)
        ]

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]


@lru_cache(maxsize=1)
def _store() -> dict[str, Chunk]:
    return {c.id: c for c in load_chunks(sorted(paths.CHUNKS.rglob("*.parquet")))}


def get_chunk(chunk_id: str) -> Chunk:
    """Full chunk with attribution. Raises KeyError for an unknown id."""
    return _store()[chunk_id]
