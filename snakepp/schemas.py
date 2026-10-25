"""Frozen interfaces shared by every module.

Do not change these without a team-approved issue: corpus, retrieval, generation and
evaluation are built against them in parallel.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["docs", "stackoverflow"]
Stage = Literal["bm25", "dense", "fused", "reranked"]


def make_chunk_id(*parts: str) -> str:
    """Stable id from source fields (e.g. url + heading path + position)."""
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


class Chunk(BaseModel):
    """One retrievable piece of text."""

    id: str
    text: str
    source: Source
    library: str  # "pandas", "numpy", "matplotlib", "scikit-learn", "scipy", "python", ...
    version: str | None = None  # "2.2.2" for docs; inferred for Stack Overflow
    version_inferred: bool = False  # True when derived from the post date
    url: str
    title: str
    score: int | None = None  # Stack Overflow answer score
    created: date | None = None
    license: str  # "CC BY-SA 4.0", "BSD-3-Clause", ...
    author: str | None = None  # Stack Overflow attribution


class Retrieved(BaseModel):
    chunk_id: str
    rank: int = Field(ge=1)
    score: float
    stage: Stage


class Task(BaseModel):
    """A benchmark problem as seen by the pipeline (never contains the reference solution)."""

    task_id: str
    benchmark: Literal["ds1000", "humaneval", "mbpp", "version_set"]
    prompt: str
    library: str | None = None
    target_versions: dict[str, str] = Field(default_factory=dict)  # {"pandas": "2.2.2"}


class Generation(BaseModel):
    task_id: str
    config: str
    prompt_hash: str
    code: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    retrieved: list[Retrieved] = Field(default_factory=list)
    raw_output: str
    latency_s: float = 0.0


class EvalResult(BaseModel):
    task_id: str
    benchmark: str
    config: str
    passed: bool
    error_type: str | None = None  # "AttributeError", "Timeout", "SyntaxError", ...
    stderr_tail: str | None = None
