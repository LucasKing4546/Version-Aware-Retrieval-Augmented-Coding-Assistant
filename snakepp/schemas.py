"""Shared types: the contract between all four lanes (ARCHITECTURE.md §4).

Frozen on contract day (Mon 16 Nov). Any change needs an issue that all four approve.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

Source = Literal["docs", "stackoverflow"]
Stage = Literal["bm25", "dense", "fused", "reranked"]
Benchmark = Literal["ds1000", "humaneval", "mbpp", "g1", "mini"]
Split = Literal["dev", "test"]

HEADING_SEP = " > "


def make_chunk_id(*parts: str) -> str:
    """Stable id from source fields: url + heading path + position."""
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


class Chunk(BaseModel):
    """One retrievable piece of text. Built by Lucas's ingest, read by everyone else."""

    id: str
    text: str
    source: Source
    library: str  # "pandas", "numpy", "matplotlib", "scikit-learn", "scipy", "python"
    version: str | None = None  # explicit for docs; inferred from the post date for Stack Overflow
    version_inferred: bool = False  # True when the version was guessed, not known
    url: str
    title: str
    heading_path: str = ""  # "page > section > subsection", joined with HEADING_SEP; gold sets key on url + this
    score: int | None = None  # Stack Overflow answer score
    created: date | None = None
    license: str  # "CC BY-SA 4.0", "BSD-3-Clause", ...
    author: str | None = None  # Stack Overflow attribution


class Retrieved(BaseModel):
    """One ranked hit. `chunk` is filled by retrieve() so consumers need no second lookup."""

    chunk_id: str
    rank: int = Field(ge=1)
    score: float
    stage: Stage
    chunk: Chunk | None = None


class Task(BaseModel):
    """A benchmark problem as the pipeline sees it. Never holds the reference solution or hidden tests."""

    task_id: str
    benchmark: Benchmark
    split: Split = "dev"
    prompt: str
    library: str | None = None
    target_versions: dict[str, str] = Field(default_factory=dict)  # {"pandas": "2.2.2"}
    sample: str | None = None  # sample input the agent may pass to run_python


class Generation(BaseModel):
    """One answer to one task under one config. Written by Enea's runs."""

    task_id: str
    config: str
    prompt_hash: str
    code: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    retrieved: list[Retrieved] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)  # agent only: {"tool", "args", "latency_s"}
    raw_output: str
    latency_s: float = 0.0
    corpus_hash: str | None = None  # from data/manifest.json


class RunResult(BaseModel):
    """Outcome of executing code. Returned by sandbox.run() and by the run_python tool."""

    stdout: str = ""
    stderr_tail: str = ""
    error_type: str | None = None  # "AttributeError", "SyntaxError", ... None when it ran cleanly
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.error_type is None and not self.timed_out


class EvalResult(BaseModel):
    """A scored generation. Written by Roberto's scorer to data/results/."""

    task_id: str
    benchmark: str
    config: str
    passed: bool
    error_type: str | None = None  # "AttributeError", "Timeout", "SyntaxError", ...
    stderr_tail: str | None = None
    corpus_hash: str | None = None


# ---- Gold sets (gold/g1.yaml, gold/g2.jsonl, gold/g3.yaml) -------------------------------------


class GoldChunkRef(BaseModel):
    """Points at a chunk by url + heading path, not by id, so gold sets don't wait for the final chunker."""

    url: str
    heading_path: str = ""

    def matches(self, chunk: Chunk) -> bool:
        return chunk.url == self.url and (not self.heading_path or chunk.heading_path == self.heading_path)


class G1Item(BaseModel):
    """Version-sensitive code task (Roberto). Tests live next to it and are never shown to the agent."""

    task_id: str
    question: str
    library: str
    target_version: str  # the test must pass here
    other_version: str  # ... and fail here
    gold_chunks: list[GoldChunkRef] = Field(default_factory=list)
    split: Split = "dev"


class G2Item(BaseModel):
    """Retrieval query with its gold chunks (Lucas)."""

    query_id: str
    query: str
    library: str
    target_versions: dict[str, str] = Field(default_factory=dict)
    gold_chunks: list[GoldChunkRef]
    origin: Literal["g1", "synthetic", "manual"] = "synthetic"
    verified: bool = False


class G3Item(BaseModel):
    """Expected agent decisions (Roberto)."""

    task_id: str
    needs_retrieval: bool
    target_versions: dict[str, str] = Field(default_factory=dict)
    source: Source | None = None
