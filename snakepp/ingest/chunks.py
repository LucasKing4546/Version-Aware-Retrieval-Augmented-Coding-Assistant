"""Reading the chunk store. Owner: Lucas."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from snakepp.schemas import Chunk


def load_chunks(paths: Iterable[str | Path]) -> Iterator[Chunk]:
    """Yield every chunk from the given parquet (or fixture .jsonl) files."""
    raise NotImplementedError("Lucas")
