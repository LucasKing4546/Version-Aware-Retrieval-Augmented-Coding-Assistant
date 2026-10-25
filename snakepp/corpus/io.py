"""Read and write chunk parquet files (data/chunks/*.parquet) with a fixed schema."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from snakepp.schemas import Chunk

CHUNKS_DIR = Path("data/chunks")

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("source", pa.string()),
    ("library", pa.string()),
    ("version", pa.string()),
    ("version_inferred", pa.bool_()),
    ("url", pa.string()),
    ("title", pa.string()),
    ("score", pa.int64()),
    ("created", pa.date32()),
    ("license", pa.string()),
    ("author", pa.string()),
])


def write_chunks(chunks: Iterable[Chunk], path: Path) -> int:
    rows = [c.model_dump() for c in chunks]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    return len(rows)


def read_chunks(paths: Path | Iterable[Path]) -> list[Chunk]:
    """Chunks from one or more parquet files, in file order; later duplicates of an id are dropped."""
    paths = [paths] if isinstance(paths, Path) else list(paths)
    seen: dict[str, Chunk] = {}
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            seen.setdefault(row["id"], Chunk.model_validate(row))
    return list(seen.values())
