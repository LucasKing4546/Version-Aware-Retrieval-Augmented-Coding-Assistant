"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from snakepp.schemas import Chunk

CHUNKS_200 = Path(__file__).parent / "fixtures" / "chunks_200.jsonl"


def load_chunks(path: Path = CHUNKS_200) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Chunk.model_validate_json(line) for line in lines if line.strip()]


@pytest.fixture(scope="session")
def chunks_200() -> list[Chunk]:
    """200 synthetic docs + Stack Overflow chunks (see scripts/make_chunk_fixture.py)."""
    return load_chunks()


@pytest.fixture(scope="session")
def chunks_by_id(chunks_200: list[Chunk]) -> dict[str, Chunk]:
    return {c.id: c for c in chunks_200}


@pytest.fixture
def chunks_parquet(tmp_path: Path, chunks_200: list[Chunk]) -> Path:
    """The fixture as a chunk parquet file, the format the indexes read from data/chunks/."""
    path = tmp_path / "chunks_fixture.parquet"
    pd.DataFrame([c.model_dump() for c in chunks_200]).to_parquet(path, index=False)
    return path
