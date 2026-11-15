"""FakeRetriever: keyword match over the chunk fixture. Owner: Kevin. Replaced by the real retrieve() on Wed 25 Nov.

Same interface as snakepp.retrieval.retriever.Retriever, so callers swap one for the other with one line:

    from tests.fakes.fake_retriever import FakeRetriever
    retriever = FakeRetriever.from_config(cfg)
    hits = retriever.retrieve("append a row to a DataFrame", "pandas", {"pandas": "2.2.2"}, k=3)

Scoring: the number of distinct query words found in the chunk's title and text. Ties go to the chunk whose version
matches the target major version, then to docs over Stack Overflow, then by id, so the output is deterministic.
It does not rerank and does not apply a version filter, penalty or drop.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from snakepp.config import Config
from snakepp.schemas import Chunk, Retrieved, Source

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "chunks_200.jsonl"

_WORD = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i in is it of on or the this to use using what when with".split()
)


def tokenize(text: str) -> set[str]:
    """Lowercase words without stopwords. A dotted name gives its parts: "DataFrame.append" -> dataframe, append."""
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


def load_fixture(path: Path = FIXTURE) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Chunk.model_validate_json(line) for line in lines if line.strip()]


def _major(version: str | None) -> str | None:
    return version.split(".")[0] if version else None


class FakeRetriever:
    def __init__(self, chunks: Iterable[Chunk], sources: Iterable[Source] = ("docs", "stackoverflow")) -> None:
        allowed = set(sources)
        self._chunks = {c.id: c for c in chunks}
        self._searchable = [c for c in self._chunks.values() if c.source in allowed]
        self._tokens = {c.id: tokenize(f"{c.title} {c.text}") for c in self._searchable}

    @classmethod
    def from_config(cls, cfg: Config, path: Path = FIXTURE) -> FakeRetriever:
        sources = [] if cfg.retrieval.mode == "none" else cfg.retrieval.sources
        return cls(load_fixture(path), sources=sources)

    def retrieve(self, query: str, library: str | None, target_versions: dict[str, str], k: int) -> list[Retrieved]:
        words = tokenize(query)
        scored = []
        for c in self._searchable:
            if library is not None and c.library != library:
                continue
            overlap = len(words & self._tokens[c.id])
            if overlap == 0:
                continue
            target = _major(target_versions.get(c.library))
            version_match = target is not None and _major(c.version) == target
            scored.append((-overlap, not version_match, c.source != "docs", c.id, c))
        scored.sort(key=lambda row: row[:4])
        return [
            Retrieved(chunk_id=c.id, rank=i, score=float(-neg), stage="bm25", chunk=c)
            for i, (neg, *_, c) in enumerate(scored[:k], start=1)
        ]

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]
