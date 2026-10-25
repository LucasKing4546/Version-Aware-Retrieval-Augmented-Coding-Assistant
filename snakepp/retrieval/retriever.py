"""Retriever: BM25 and/or dense -> RRF -> rerank -> version filter -> top-k, as set by RetrievalConfig."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from snakepp.config import Config
from snakepp.corpus.io import CHUNKS_DIR, read_chunks
from snakepp.retrieval import version_filter
from snakepp.retrieval.bm25 import BM25Index
from snakepp.retrieval.dense import DenseIndex, Encoder
from snakepp.retrieval.fusion import rrf
from snakepp.retrieval.rerank import Reranker, Scorer
from snakepp.schemas import Chunk, Retrieved


def corpus_files(cfg: Config, root: Path = CHUNKS_DIR) -> list[Path]:
    """docs_*.parquet for docs; so_clean.parquet or so_leaky.parquet for Stack Overflow."""
    files: list[Path] = []
    if "docs" in cfg.retrieval.sources:
        files += sorted(root.glob("docs_*.parquet"))
    if "stackoverflow" in cfg.retrieval.sources:
        so = root / f"so_{cfg.retrieval.so_index}.parquet"
        files += [so] if so.exists() else []
    return files


class Retriever:
    def __init__(
        self, cfg: Config, chunks: Iterable[Chunk], encoder: Encoder | None = None, scorer: Scorer | None = None
    ) -> None:
        r = cfg.retrieval
        self.cfg = cfg
        self.chunks = {c.id: c for c in chunks if c.source in r.sources}
        if not self.chunks:
            raise ValueError(f"no chunks for sources {r.sources}: build the corpus into data/chunks first")
        items = list(self.chunks.values())
        self.bm25 = BM25Index(items) if r.bm25 else None
        self.dense = DenseIndex(items, r.dense_model, encoder) if r.dense else None
        self.reranker = Reranker(r.rerank_model, scorer) if r.rerank else None

    @classmethod
    def from_disk(cls, cfg: Config, root: Path = CHUNKS_DIR, **kwargs) -> Retriever:
        return cls(cfg, read_chunks(corpus_files(cfg, root)), **kwargs)

    def candidates(self, query: str) -> list[Retrieved]:
        n = self.cfg.retrieval.candidates
        lists = [idx.search(query, n) for idx in (self.bm25, self.dense) if idx is not None]
        if len(lists) == 1:
            return lists[0]
        return rrf(lists, k=self.cfg.retrieval.rrf_k, top_n=n)

    def retrieve(self, query: str, target_versions: dict[str, str] | None = None) -> list[Retrieved]:
        results = self.candidates(query)
        if self.reranker is not None:
            results = self.reranker.rerank(query, results, self.chunks)
        vf = self.cfg.version_filter
        targets = version_filter.detect_versions(query, target_versions)
        results = version_filter.apply(results, self.chunks, targets, vf.mode, vf.penalty, vf.max_minor_gap)
        return results[: self.cfg.retrieval.top_k]
