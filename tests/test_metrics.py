"""Retrieval metrics: the formulas on hand-made hits, then evaluate() on the real fixture with a small gold set."""

from pathlib import Path

import pytest

from snakepp.config import Config, RetrievalConfig, VersionFilterConfig
from snakepp.ingest.chunks import load_chunks
from snakepp.retrieval.metrics import evaluate, load_g2, recall_at_k, reciprocal_rank, version_correct_at_k
from snakepp.retrieval.retriever import Retriever
from snakepp.schemas import Chunk, GoldChunkRef
from tests.fakes.fake_encoder import HashingEncoder

FIXTURES = Path(__file__).parent / "fixtures"


def chunk(url: str, heading: str = "", version: str | None = "2.2.3") -> Chunk:
    return Chunk(id=url + heading, text="t", source="docs", library="pandas", version=version, url=url,
                 title="t", heading_path=heading, license="BSD-3-Clause")  # fmt: skip


A, B, C = chunk("a"), chunk("b", "x > y"), chunk("c", version="1.5.3")
GOLD = [GoldChunkRef(url="a"), GoldChunkRef(url="b", heading_path="x > y")]


def test_recall_at_k():
    assert recall_at_k([C, A, B], GOLD, k=1) == 0.0
    assert recall_at_k([C, A, B], GOLD, k=2) == 0.5
    assert recall_at_k([C, A, B], GOLD, k=3) == 1.0
    assert recall_at_k([chunk("b", "x > other")], GOLD, k=1) == 0.0  # heading path must match when given
    with pytest.raises(ValueError):
        recall_at_k([A], [], k=1)


def test_reciprocal_rank():
    assert reciprocal_rank([C, B, A], GOLD) == 0.5
    assert reciprocal_rank([C], GOLD) == 0.0


def test_version_correct_counts_missing_slots_as_wrong():
    target = {"pandas": "2.2.3"}
    assert version_correct_at_k([A, C, B], target, k=3) == pytest.approx(2 / 3)
    assert version_correct_at_k([A], target, k=3) == pytest.approx(1 / 3)
    assert version_correct_at_k([C, chunk("d", version=None)], target, k=2) == 0.5  # unknown version is fine
    assert version_correct_at_k([C], {}, k=1) == 1.0  # no target: nothing can mismatch


@pytest.fixture(scope="module")
def real_chunks() -> list[Chunk]:
    return list(load_chunks(FIXTURES / "fixture_chunks.jsonl"))


def bm25(chunks, mode: str) -> Retriever:
    cfg = Config(name=mode, retrieval=RetrievalConfig(mode="bm25", rerank=False),
                 version_filter=VersionFilterConfig(mode=mode))  # fmt: skip
    return Retriever(chunks, cfg, encode=HashingEncoder(), cache_dir=None)


def test_evaluate_on_the_fixture_gold_set(real_chunks):
    items = load_g2(FIXTURES / "g2_fixture.jsonl")
    urls = {c.url for c in real_chunks}
    assert len(items) == 5 and all(ref.url in urls for item in items for ref in item.gold_chunks)

    off = evaluate(bm25(real_chunks, "off"), items, ks=(1, 5, 10))
    hard = evaluate(bm25(real_chunks, "hard"), items, ks=(1, 5, 10))
    assert set(off) == {"mrr", "recall@1", "recall@5", "recall@10",
                        "version_correct@1", "version_correct@5", "version_correct@10"}  # fmt: skip
    assert all(0.0 <= v <= 1.0 for v in off.values())
    assert off["recall@1"] <= off["recall@5"] <= off["recall@10"]
    assert hard["version_correct@5"] > off["version_correct@5"]  # the filter's whole point
    assert hard["mrr"] >= off["mrr"]  # gold chunks are 2.2.3, so dropping 1.5.3 can only help them
