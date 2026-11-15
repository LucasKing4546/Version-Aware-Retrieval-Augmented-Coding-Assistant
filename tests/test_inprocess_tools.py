"""InProcessTools over stand-ins (the real retriever and sandbox are not built yet)."""

from __future__ import annotations

import re

import pytest

from snakepp.config import load_config
from snakepp.generation.inprocess_tools import InProcessTools
from snakepp.generation.tools import Tools
from snakepp.retrieval import retriever
from snakepp.schemas import Chunk, Retrieved, RunResult


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class KeywordRetriever:
    """Minimal stand-in for Retriever.retrieve: library match, then token overlap, best first.

    Replace with tests/fakes FakeRetriever (Kevin) once it lands.
    """

    def __init__(self, chunks: list[Chunk], fill_chunk: bool = True) -> None:
        self.chunks = chunks
        self.fill_chunk = fill_chunk
        self.calls: list[tuple[str, str | None, dict[str, str], int]] = []

    def retrieve(self, query: str, library: str | None, target_versions: dict[str, str], k: int) -> list[Retrieved]:
        self.calls.append((query, library, target_versions, k))
        words = _tokens(query)
        scored = [(len(words & _tokens(c.text)), c) for c in self.chunks if library is None or c.library == library]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda sc: (-sc[0], sc[1].id))
        return [
            Retrieved(
                chunk_id=c.id,
                rank=i,
                score=float(s),
                stage="reranked",
                chunk=c if self.fill_chunk else None,
            )
            for i, (s, c) in enumerate(scored[:k], start=1)
        ]


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs) -> RunResult:
        self.calls.append((args, kwargs))
        return RunResult(stdout="ok\n")


def make_tools(chunks_200, chunks_by_id, retrievers=None, run=None) -> InProcessTools:
    if retrievers is None:
        docs = [c for c in chunks_200 if c.source == "docs"]
        so = [c for c in chunks_200 if c.source == "stackoverflow"]
        retrievers = {"docs": KeywordRetriever(docs), "stackoverflow": KeywordRetriever(so)}

    def get_chunk(chunk_id: str) -> Chunk:
        return chunks_by_id[chunk_id]

    return InProcessTools(
        retrievers=retrievers,
        get_chunk=get_chunk,
        detect_versions=lambda text: {"pandas": "1.5.3"} if "1.5.3" in text else {},
        list_indexed_versions=lambda library: {"pandas": ["1.5.3", "2.2.2"]},
        run=run or RecordingRunner(),
    )


def test_implements_the_tools_protocol(chunks_200, chunks_by_id):
    assert isinstance(make_tools(chunks_200, chunks_by_id), Tools)


@pytest.mark.parametrize("tool,source", [("search_docs", "docs"), ("search_stackoverflow", "stackoverflow")])
def test_search_returns_only_its_source_with_chunks_and_ranks(chunks_200, chunks_by_id, tool, source):
    hits = getattr(make_tools(chunks_200, chunks_by_id), tool)("concat append dataframe", "pandas", top_k=4)
    assert hits, "the fixture has pandas chunks about concat"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert len(hits) <= 4
    assert all(h.chunk is not None and h.chunk.source == source and h.chunk.library == "pandas" for h in hits)


def test_stackoverflow_hits_keep_attribution(chunks_200, chunks_by_id):
    hits = make_tools(chunks_200, chunks_by_id).search_stackoverflow("concat dataframe", "pandas")
    assert hits and all(h.chunk.author and h.chunk.license for h in hits)


def test_version_becomes_target_versions(chunks_200, chunks_by_id):
    docs = KeywordRetriever(chunks_200)
    tools = make_tools(chunks_200, chunks_by_id, retrievers={"docs": docs})
    tools.search_docs("concat", "Pandas ", version="1.5.3", top_k=3)
    tools.search_docs("concat", "pandas")
    assert docs.calls == [("concat", "pandas", {"pandas": "1.5.3"}, 3), ("concat", "pandas", {}, 5)]


def test_shared_backend_is_split_by_source_and_reranked(chunks_200, chunks_by_id):
    mixed = KeywordRetriever(chunks_200)
    tools = make_tools(chunks_200, chunks_by_id, retrievers={"docs": mixed, "stackoverflow": mixed})
    so = tools.search_stackoverflow("concat dataframe append", "pandas", top_k=5)
    assert so and all(h.chunk.source == "stackoverflow" for h in so)
    assert [h.rank for h in so] == list(range(1, len(so) + 1))


def test_missing_chunk_is_looked_up(chunks_200, chunks_by_id):
    bare = KeywordRetriever([c for c in chunks_200 if c.source == "docs"], fill_chunk=False)
    hits = make_tools(chunks_200, chunks_by_id, retrievers={"docs": bare}).search_docs("concat", "pandas")
    assert hits and all(h.chunk == chunks_by_id[h.chunk_id] for h in hits)


def test_disabled_source_returns_nothing(chunks_200, chunks_by_id):
    tools = make_tools(chunks_200, chunks_by_id, retrievers={"docs": KeywordRetriever(chunks_200)})
    assert tools.search_stackoverflow("concat", "pandas") == []
    assert make_tools(chunks_200, chunks_by_id, retrievers={}).search_docs("concat", "pandas") == []


def test_rejects_non_positive_top_k(chunks_200, chunks_by_id):
    with pytest.raises(ValueError, match="top_k"):
        make_tools(chunks_200, chunks_by_id).search_docs("concat", "pandas", top_k=0)


def test_run_python_never_forwards_a_test(chunks_200, chunks_by_id):
    runner = RecordingRunner()
    result = make_tools(chunks_200, chunks_by_id, run=runner).run_python("print(1)", {"pandas": "2.2.2"}, sample="x=1")
    assert result.ok
    assert runner.calls == [(("print(1)", {"pandas": "2.2.2"}), {"sample": "x=1"})]


def test_delegating_tools(chunks_200, chunks_by_id):
    tools = make_tools(chunks_200, chunks_by_id)
    assert tools.detect_versions("pandas==1.5.3") == {"pandas": "1.5.3"}
    assert tools.list_indexed_versions("pandas") == {"pandas": ["1.5.3", "2.2.2"]}
    first = chunks_200[0]
    assert tools.get_chunk(first.id) == first
    with pytest.raises(KeyError):
        tools.get_chunk("no-such-id")


@pytest.mark.parametrize(
    "config,expected",
    [
        ("c0_none", []),
        ("c1_docs", [["docs"]]),
        ("c2_so", [["stackoverflow"]]),
        ("c5_full", [["docs"], ["stackoverflow"]]),
    ],
)
def test_from_config_builds_one_retriever_per_enabled_source(monkeypatch, config, expected):
    seen = []

    def fake_from_config(cfg):
        seen.append(cfg)
        return KeywordRetriever([])

    monkeypatch.setattr(retriever.Retriever, "from_config", staticmethod(fake_from_config))
    cfg = load_config(config)
    tools = InProcessTools.from_config(cfg)

    assert [c.retrieval.sources for c in seen] == expected
    # Everything except the source list is passed through unchanged.
    for c in seen:
        assert c.model_dump(exclude={"retrieval": {"sources"}}) == cfg.model_dump(exclude={"retrieval": {"sources"}})
    assert isinstance(tools, Tools)
