"""Tiny disk cache for slow steps (embeddings, LLM calls, sandbox runs)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

CACHE_DIR = Path("data/cache")


def key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cached_json(namespace: str, cache_key: str, compute: Callable[[], Any], root: Path = CACHE_DIR) -> Any:
    """Return the cached JSON value for (namespace, key), computing and storing it if missing."""
    path = root / namespace / f"{cache_key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=str), encoding="utf-8")
    return value
