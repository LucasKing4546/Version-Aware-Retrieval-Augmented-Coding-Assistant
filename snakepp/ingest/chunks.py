"""Reading and writing the chunk store. Owner: Lucas."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from snakepp.schemas import Chunk

# Explicit types, so an all-null column (score/created/author on docs) stays int/date/string and files from
# different sources concatenate cleanly.
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("source", pa.string()),
        ("library", pa.string()),
        ("version", pa.string()),
        ("version_inferred", pa.bool_()),
        ("url", pa.string()),
        ("title", pa.string()),
        ("heading_path", pa.string()),
        ("score", pa.int64()),
        ("created", pa.date32()),
        ("license", pa.string()),
        ("author", pa.string()),
    ]
)


def load_chunks(paths: Iterable[str | Path]) -> Iterator[Chunk]:
    """Yield every chunk from the given parquet (or fixture .jsonl) files."""
    if isinstance(paths, str | Path):  # a lone path would otherwise iterate character by character
        paths = [paths]
    for path in map(Path, paths):
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                yield from (Chunk.model_validate_json(line) for line in f if line.strip())
        else:
            yield from (Chunk.model_validate(row) for row in pq.read_table(path).to_pylist())


def write_chunks(chunks: Iterable[Chunk], path: Path) -> Path:
    """Write chunks to one parquet file (replacing it), creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([c.model_dump() for c in chunks], schema=SCHEMA), path)
    return path
