import zlib

import numpy as np
import pytest

from snakepp import cache
from snakepp.config import Config
from snakepp.corpus.io import write_chunks
from snakepp.retrieval import Retriever, corpus_files
from snakepp.retrieval.bm25 import tokenize
from snakepp.retrieval.dense import DenseIndex
from snakepp.retrieval.rerank import Reranker
from snakepp.retrieval.version_filter import detect_versions, versions_in_text


def bag_of_words(texts):
    """Deterministic fake embedding: hashed bag of identifier tokens, L2-normalized."""
    out = np.zeros((len(texts), 256), dtype="float32")
    for row, text in enumerate(texts):
        for tok in tokenize(text):
            out[row, zlib.crc32(tok.encode()) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def overlap_scorer(pairs):
    return [float(len(set(tokenize(q)) & set(tokenize(p)))) - 2 for q, p in pairs]


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")


def _cfg(**retrieval):
    return Config.model_validate({"name": "t", "retrieval": retrieval})


def test_dense_index_and_cache(chunks_200, chunks_by_id):
    calls = []

    def encoder(texts):
        calls.append(len(texts))
        return bag_of_words(texts)

    index = DenseIndex(chunks_200, "fake", encoder)
    hits = index.search("pandas.concat ignore_index", k=3)
    assert hits[0].stage == "dense" and "concat" in chunks_by_id[hits[0].chunk_id].title
    DenseIndex(chunks_200, "fake", encoder)  # embeddings come from data/cache/dense
    assert calls.count(200) == 1


def test_reranker_sigmoid_and_order(chunks_200, chunks_by_id):
    cands = DenseIndex(chunks_200, "fake", bag_of_words).search("numpy isin membership test", k=10)
    out = Reranker("fake", overlap_scorer).rerank("numpy isin membership test", cands, chunks_by_id)
    assert out[0].stage == "reranked" and all(0 < r.score < 1 for r in out)
    assert [r.score for r in out] == sorted((r.score for r in out), reverse=True)


def test_versions_in_text():
    assert versions_in_text("I use pandas 1.5 and sklearn==1.4.0") == {"pandas": "1.5", "scikit-learn": "1.4.0"}
    assert versions_in_text("read a csv with pandas") == {}
    assert detect_versions("pandas 1.5", {"pandas": "2.2.2"}) == {"pandas": "2.2.2"}


def test_retriever_full_pipeline(chunks_200, chunks_by_id):
    r = Retriever(_cfg(top_k=3), chunks_200, encoder=bag_of_words, scorer=overlap_scorer)
    hits = r.retrieve("How to append a row to a DataFrame?", {"pandas": "2.2.2"})
    assert len(hits) == 3 and hits[0].stage == "reranked"


def test_retriever_hard_filter_drops_old_versions(chunks_200, chunks_by_id):
    cfg = Config.model_validate({"name": "t", "retrieval": {"top_k": 10, "rerank": False},
                                 "version_filter": {"mode": "hard"}})
    hits = Retriever(cfg, chunks_200, encoder=bag_of_words).retrieve("DataFrame.append", {"pandas": "2.2.2"})
    assert hits and all(not (chunks_by_id[h.chunk_id].version or "").startswith("1.") for h in hits
                        if chunks_by_id[h.chunk_id].library == "pandas")


def test_retriever_sources_and_bm25_only(chunks_200, chunks_by_id):
    r = Retriever(_cfg(sources=["stackoverflow"], dense=False, rerank=False), chunks_200)
    hits = r.retrieve("matplotlib colormaps viridis")
    assert hits and all(chunks_by_id[h.chunk_id].source == "stackoverflow" for h in hits)
    assert hits[0].stage == "bm25"


def test_corpus_files(tmp_path, chunks_200):
    write_chunks(chunks_200[:5], tmp_path / "docs_pandas_2.2.2.parquet")
    write_chunks(chunks_200[-5:], tmp_path / "so_clean.parquet")
    assert [p.name for p in corpus_files(_cfg(), tmp_path)] == ["docs_pandas_2.2.2.parquet", "so_clean.parquet"]
    assert [p.name for p in corpus_files(_cfg(so_index="leaky"), tmp_path)] == ["docs_pandas_2.2.2.parquet"]
    with pytest.raises(ValueError):
        Retriever(_cfg(sources=["docs"]), [c for c in chunks_200 if c.source == "stackoverflow"])


def test_minor_gap(chunks_200):
    from snakepp.retrieval.version_filter import compatible

    old_mpl = next(c for c in chunks_200 if c.library == "matplotlib" and c.version == "3.7.0")
    assert compatible(old_mpl, {"matplotlib": "3.9.0"})
    assert not compatible(old_mpl, {"matplotlib": "3.9.0"}, max_minor_gap=1)
    assert compatible(old_mpl, {"matplotlib": "3.8.4"}, max_minor_gap=1)
