from pathlib import Path

import pytest

from snakepp.corpus.io import read_chunks, write_chunks
from snakepp.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_keeps_identifiers_and_parts():
    toks = tokenize("How do I use df.groupby() with numeric_only in pandas.DataFrame?")
    assert {"df.groupby", "groupby", "numeric_only", "numeric", "only", "pandas.dataframe", "data", "frame"} <= set(
        toks
    )
    assert "how" not in toks and "do" not in toks


def test_tokenize_camel_case():
    assert tokenize("OneHotEncoder") == ["onehotencoder", "one", "hot", "encoder"]
    assert tokenize("HTTPServer") == ["httpserver", "http", "server"]


def test_bm25_finds_api_by_name(chunks_200, chunks_by_id):
    index = BM25Index(chunks_200)
    hits = index.search("DataFrame.append rows", k=5)
    assert hits[0].stage == "bm25" and hits[0].rank == 1
    assert "DataFrame.append" in chunks_by_id[hits[0].chunk_id].title
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_bm25_plain_words(chunks_200, chunks_by_id):
    hits = BM25Index(chunks_200).search("one hot encode categorical features", k=3)
    assert any("OneHotEncoder" in chunks_by_id[h.chunk_id].title for h in hits)


def test_bm25_rejects_empty():
    with pytest.raises(ValueError):
        BM25Index([])


def test_bm25_unknown_query(chunks_200):
    assert BM25Index(chunks_200).search("zzzqqq", k=5) == []


def test_parquet_roundtrip(tmp_path: Path, chunks_200):
    path = tmp_path / "c.parquet"
    assert write_chunks(chunks_200, path) == 200
    assert read_chunks(path) == chunks_200
    assert read_chunks([path, path]) == chunks_200  # duplicates dropped
