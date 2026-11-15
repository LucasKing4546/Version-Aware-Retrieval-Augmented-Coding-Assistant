"""BM25, dense, RRF and Retriever on Lucas's real-data fixture (pandas 1.5.3 / 2.2.3 + Stack Overflow)."""

import os
from pathlib import Path

import pytest

from snakepp.config import Config, RetrievalConfig, VersionFilterConfig, load_config
from snakepp.ingest.chunks import load_chunks
from snakepp.retrieval.bm25 import BM25Index
from snakepp.retrieval.dense import DenseIndex, embed_cached
from snakepp.retrieval.fusion import rrf
from snakepp.retrieval.retriever import Retriever
from snakepp.retrieval.tokenizer import tokenize
from snakepp.schemas import Chunk
from tests.fakes.fake_encoder import HashingEncoder

FIXTURE = Path(__file__).parent / "fixtures" / "fixture_chunks.jsonl"


@pytest.fixture(scope="module")
def real_chunks() -> list[Chunk]:
    return list(load_chunks(FIXTURE))


def cfg(mode: str = "hybrid", **retrieval) -> Config:
    return Config(name=f"t_{mode}", retrieval=RetrievalConfig(mode=mode, rerank=False, **retrieval))


def make(chunks, mode="hybrid", tmp_path=None, encoder=None, **retrieval) -> Retriever:
    return Retriever(chunks, cfg(mode, **retrieval), encode=encoder or HashingEncoder(), cache_dir=tmp_path)


# ---- tokenizer -------------------------------------------------------------------------------------------


def test_tokenizer_keeps_dotted_names_and_their_pieces():
    assert tokenize("How do I use pd.DataFrame.append()?") == [
        "pd.dataframe.append", "pd", "dataframe", "data", "frame", "append",
    ]  # fmt: skip


def test_tokenizer_snake_case_versions_and_stopwords():
    assert tokenize("is_monotonic in pandas 2.2.3") == ["is_monotonic", "monotonic", "pandas", "2.2.3"]
    assert tokenize("the of and") == []


# ---- BM25 ------------------------------------------------------------------------------------------------


def test_bm25_finds_the_api_page_by_name(real_chunks):
    hits = BM25Index(real_chunks).search("DataFrame.append", n=5)
    top = {c.id: c for c in real_chunks}[hits[0][0]]
    assert "DataFrame.append" in top.url
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_bm25_library_filter_and_empty_query(real_chunks):
    index = BM25Index(real_chunks)
    assert index.search("append", n=5, library="numpy") == []
    assert index.search("the of", n=5) == []


# ---- RRF -------------------------------------------------------------------------------------------------


def test_rrf_rewards_agreement_and_breaks_ties_by_id():
    fused = rrf([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert [cid for cid, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert [cid for cid, _ in rrf([["y"], ["x"]])] == ["x", "y"]


# ---- dense -----------------------------------------------------------------------------------------------


def test_dense_search_and_library_selector(real_chunks, tmp_path):
    index = DenseIndex(real_chunks, HashingEncoder(), "fake", cache_dir=tmp_path)
    hits = index.search("DataFrame.applymap elementwise", n=500)
    assert len(hits) == len(real_chunks)  # n larger than the index is clamped
    assert index.search("applymap", n=5, library="numpy") == []
    assert len(index.search("applymap", n=5, library="pandas")) == 5


def test_embedding_cache_encodes_each_text_once(real_chunks, tmp_path):
    enc = HashingEncoder()
    first = embed_cached(real_chunks[:10], enc, "fake/model", tmp_path)
    again = embed_cached(real_chunks[:12], enc, "fake/model", tmp_path)
    assert enc.encoded == 12  # 10, then only the 2 new ones
    assert (again[:10] == first).all()
    assert list(tmp_path.glob("*.npz")) == [tmp_path / "fake_model.npz"]


# ---- Retriever -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode,stage", [("bm25", "bm25"), ("dense", "dense"), ("hybrid", "fused")])
def test_retrieve_contract(real_chunks, tmp_path, mode, stage):
    hits = make(real_chunks, mode, tmp_path).retrieve("append rows to a DataFrame", "pandas", {}, k=5)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(h.stage == stage and h.chunk is not None and h.chunk.id == h.chunk_id for h in hits)


def test_hybrid_finds_both_sides_of_the_append_trap(real_chunks, tmp_path):
    hits = make(real_chunks, "hybrid", tmp_path).retrieve("DataFrame.append concat", "pandas", {}, k=10)
    versions = {h.chunk.version for h in hits if h.chunk.source == "docs"}
    assert {"1.5.3", "2.2.3"} <= versions


def test_stage_lists_expose_every_stage(real_chunks, tmp_path):
    lists = make(real_chunks, "hybrid", tmp_path).stage_lists("iteritems", "pandas", n=20)
    assert set(lists) == {"bm25", "dense", "fused"}
    assert {cid for cid, _ in lists["fused"]} == {cid for name in ("bm25", "dense") for cid, _ in lists[name]}


def test_sources_and_none_mode(real_chunks, tmp_path):
    docs = make(real_chunks, "bm25", tmp_path, sources=["docs"]).retrieve("append", None, {}, k=20)
    assert docs and all(h.chunk.source == "docs" for h in docs)
    assert make(real_chunks, "none", tmp_path).retrieve("append", None, {}, k=5) == []


def test_get_chunk(real_chunks, tmp_path):
    r = make(real_chunks, "bm25", tmp_path)
    assert r.get_chunk(real_chunks[0].id) == real_chunks[0]
    with pytest.raises(KeyError):
        r.get_chunk("missing")


@pytest.mark.parametrize(
    "config",
    [
        Config(name="rerank"),
        Config(name="vf", retrieval=RetrievalConfig(rerank=False), version_filter=VersionFilterConfig(mode="soft")),
        Config(name="leaky", retrieval=RetrievalConfig(rerank=False, corpus="leaky")),
    ],
)
def test_week9_features_fail_loudly(real_chunks, config):
    with pytest.raises(NotImplementedError, match="Week 9"):
        Retriever(real_chunks, config)


def test_week8_configs_are_supported():
    for name in ("c0_none", "c3_dense", "c4_hybrid"):
        Retriever([], load_config(name))  # no chunks: builds nothing, but the config is accepted


@pytest.mark.skipif(not os.environ.get("SNAKEPP_SLOW"), reason="set SNAKEPP_SLOW=1 to run with the real bge-small")
def test_real_embedder_end_to_end(real_chunks, tmp_path):
    r = Retriever(real_chunks, cfg("hybrid"), cache_dir=tmp_path)
    hits = r.retrieve("How do I append a row to a DataFrame?", "pandas", {"pandas": "2.2.3"}, k=5)
    # Stack Overflow urls carry only a question id, so check the text: every top-5 hit should be on topic.
    assert all("append" in h.chunk.text.lower() or "concat" in h.chunk.text.lower() for h in hits)
    assert any(h.chunk.source == "docs" for h in hits)
