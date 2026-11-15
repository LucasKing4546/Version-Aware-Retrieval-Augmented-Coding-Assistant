"""Folder layout agreed on contract day. Build paths through these helpers, never by hand.

data/chunks/<source>/<library>/<version>.parquet
data/manifest.json
data/index/
data/cache/
data/results/<benchmark>/<config>_<YYYY-MM-DD>.jsonl   (append-only, never overwritten)
gold/g1.yaml, gold/g2.jsonl, gold/g3.yaml
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHUNKS = DATA / "chunks"
MANIFEST = DATA / "manifest.json"
INDEX = DATA / "index"
CACHE = DATA / "cache"
RESULTS = DATA / "results"
GOLD = ROOT / "gold"
G1 = GOLD / "g1.yaml"
G2 = GOLD / "g2.jsonl"
G3 = GOLD / "g3.yaml"

UNVERSIONED = "unversioned"  # file name for Stack Overflow chunks whose version could not be inferred


def chunk_path(source: str, library: str, version: str | None, root: Path = CHUNKS) -> Path:
    return root / source / library / f"{version or UNVERSIONED}.parquet"


def results_path(benchmark: str, config: str, day: date | None = None, root: Path = RESULTS) -> Path:
    day = day or date.today()
    return root / benchmark / f"{config}_{day.isoformat()}.jsonl"
