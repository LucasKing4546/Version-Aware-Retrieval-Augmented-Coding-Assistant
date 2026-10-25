import importlib.util
import sys
from pathlib import Path

import pandas as pd
from conftest import load_chunks

from snakepp.retrieval.version_filter import compatible
from snakepp.schemas import Chunk


def _script():
    spec = importlib.util.spec_from_file_location("make_chunk_fixture", Path("scripts/make_chunk_fixture.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while building Api
    spec.loader.exec_module(module)
    return module


def test_fixture_matches_generator(chunks_200):
    assert _script().build_chunks() == chunks_200


def test_write_jsonl_roundtrip(tmp_path, chunks_200):
    out = tmp_path / "c.jsonl"
    _script().write_jsonl(chunks_200[:5], out)
    assert load_chunks(out) == chunks_200[:5]


def test_fixture_shape(chunks_200):
    assert len(chunks_200) == 200
    assert len({c.id for c in chunks_200}) == 200
    assert {c.source for c in chunks_200} == {"docs", "stackoverflow"}
    assert {"pandas", "numpy", "matplotlib", "scikit-learn", "scipy", "python"} <= {c.library for c in chunks_200}


def test_stackoverflow_chunks_keep_attribution(chunks_200):
    so = [c for c in chunks_200 if c.source == "stackoverflow"]
    assert all(c.author and c.license == "CC BY-SA 4.0" and c.created for c in so)
    assert any(c.version is None for c in so) and any(c.version_inferred for c in so)


def test_fixture_exercises_version_filter(chunks_200):
    target = {"pandas": "2.2.2"}
    pandas_docs = [c for c in chunks_200 if c.library == "pandas" and c.source == "docs"]
    assert any(compatible(c, target) for c in pandas_docs)
    assert any(not compatible(c, target) for c in pandas_docs)
    assert any("DataFrame.append" in c.text and c.version == "1.5.3" for c in pandas_docs)


def test_chunks_parquet(chunks_parquet, chunks_by_id):
    df = pd.read_parquet(chunks_parquet)
    assert len(df) == 200 and set(df["id"]) == set(chunks_by_id)
    row = df.iloc[0].to_dict()
    row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    assert Chunk.model_validate(row).id == row["id"]
