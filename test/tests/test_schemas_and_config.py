from pathlib import Path

from snakepp.config import load_config
from snakepp.schemas import Chunk, make_chunk_id


def test_chunk_roundtrip():
    c = Chunk(id=make_chunk_id("u", "h"), text="x", source="docs", library="pandas", version="2.2.2",
              url="https://pandas.pydata.org", title="concat", license="BSD-3-Clause")
    assert Chunk.model_validate_json(c.model_dump_json()) == c


def test_chunk_id_stable():
    assert make_chunk_id("a", "b") == make_chunk_id("a", "b") != make_chunk_id("a", "c")


def test_all_configs_load():
    names = {load_config(p).name for p in Path("configs").glob("*.yaml")}
    assert {"c0_none", "c5_full", "c5_full_vf"} <= names
    assert load_config("configs/c0_none.yaml").retrieval.enabled is False
