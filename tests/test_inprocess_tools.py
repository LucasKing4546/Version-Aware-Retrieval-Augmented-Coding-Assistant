"""InProcessTools over the fakes (FakeRetriever, a recording runner)."""

from __future__ import annotations

import pytest

from snakepp.config import load_config
from snakepp.generation.inprocess_tools import InProcessTools
from snakepp.generation.tools import Tools
from snakepp.retrieval import retriever
from snakepp.schemas import Chunk, Retrieved, RunResult
from tests.fakes.fake_retriever import FakeRetriever


class SpyRetriever:
    """Wraps a retriever: records each call and can drop `chunk` from the hits."""

    def __init__(self, inner: FakeRetriever, drop_chunk: bool = False) -> None:
        self.inner = inner
        self.drop_chunk = drop_chunk
        self.calls: list[tuple[str, str | None, dict[str, str], int]] = []

    def retrieve(self, query: str, library: str | None, target_versions: dict[str, str], k: int) -> list[Retrieved]:
        self.calls.append((query, library, target_versions, k))
        hits = self.inner.retrieve(query, library, target_versions, k)
        return [h.model_copy(update={"chunk": None}) for h in hits] if self.drop_chunk else hits


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs) -> RunResult:
        self.calls.append((args, kwargs))
        return RunResult(stdout="ok\n")


def per_source(chunks: list[Chunk]) -> dict[str, FakeRetriever]:
    return {source: FakeRetriever(chunks, sources=[source]) for source in ("docs", "stackoverflow")}


def make_tools(chunks: list[Chunk], retrievers=None, run=None) -> InProcessTools:
    return InProcessTools(
        retrievers=per_source(chunks) if retrievers is None else retrievers,
        get_chunk=FakeRetriever(chunks).get_chunk,
        detect_versions=lambda text: {"pandas": "1.5.3"} if "1.5.3" in text else {},
        list_indexed_versions=lambda library: {"pandas": ["1.5.3", "2.2.2"]},
        run=run or RecordingRunner(),
    )


def test_implements_the_tools_protocol(chunks_200):
    assert isinstance(make_tools(chunks_200), Tools)


@pytest.mark.parametrize("tool,source", [("search_docs", "docs"), ("search_stackoverflow", "stackoverflow")])
def test_search_returns_only_its_source_with_chunks_and_ranks(chunks_200, tool, source):
    hits = getattr(make_tools(chunks_200), tool)("append a row to a DataFrame", "pandas", top_k=4)
    assert hits, "the fixture has pandas chunks about DataFrame.append"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert len(hits) <= 4
    assert all(h.chunk is not None and h.chunk.source == source and h.chunk.library == "pandas" for h in hits)


def test_stackoverflow_hits_keep_attribution(chunks_200):
    hits = make_tools(chunks_200).search_stackoverflow("concat dataframe", "pandas")
    assert hits and all(h.chunk.author and h.chunk.license for h in hits)


def test_version_becomes_target_versions(chunks_200):
    docs = SpyRetriever(FakeRetriever(chunks_200, sources=["docs"]))
    tools = make_tools(chunks_200, retrievers={"docs": docs})
    tools.search_docs("concat", "Pandas ", version="1.5.3", top_k=3)
    tools.search_docs("concat", "pandas")
    assert docs.calls == [("concat", "pandas", {"pandas": "1.5.3"}, 3), ("concat", "pandas", {}, 5)]


def test_target_version_reaches_the_ranking(chunks_200):
    # FakeRetriever breaks ties toward the target major version, so the version must get through.
    tools = make_tools(chunks_200)
    old = tools.search_docs("pandas concat", "pandas", version="1.5.3", top_k=1)
    new = tools.search_docs("pandas concat", "pandas", version="2.2.2", top_k=1)
    assert old[0].chunk.version.startswith("1.") and new[0].chunk.version.startswith("2.")


def test_shared_backend_is_split_by_source_and_reranked(chunks_200):
    mixed = FakeRetriever(chunks_200)
    tools = make_tools(chunks_200, retrievers={"docs": mixed, "stackoverflow": mixed})
    so = tools.search_stackoverflow("concat dataframe append", "pandas", top_k=5)
    assert so and all(h.chunk.source == "stackoverflow" for h in so)
    assert [h.rank for h in so] == list(range(1, len(so) + 1))


def test_missing_chunk_is_looked_up(chunks_200, chunks_by_id):
    bare = SpyRetriever(FakeRetriever(chunks_200, sources=["docs"]), drop_chunk=True)
    hits = make_tools(chunks_200, retrievers={"docs": bare}).search_docs("concat", "pandas")
    assert hits and all(h.chunk == chunks_by_id[h.chunk_id] for h in hits)


def test_disabled_source_returns_nothing(chunks_200):
    tools = make_tools(chunks_200, retrievers={"docs": FakeRetriever(chunks_200, sources=["docs"])})
    assert tools.search_stackoverflow("concat", "pandas") == []
    assert make_tools(chunks_200, retrievers={}).search_docs("concat", "pandas") == []


def test_rejects_non_positive_top_k(chunks_200):
    with pytest.raises(ValueError, match="top_k"):
        make_tools(chunks_200).search_docs("concat", "pandas", top_k=0)


def test_run_python_never_forwards_a_test(chunks_200):
    runner = RecordingRunner()
    result = make_tools(chunks_200, run=runner).run_python("print(1)", {"pandas": "2.2.2"}, sample="x=1")
    assert result.ok
    assert runner.calls == [(("print(1)", {"pandas": "2.2.2"}), {"sample": "x=1"})]


def test_delegating_tools(chunks_200):
    tools = make_tools(chunks_200)
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

    def spy_from_config(cfg):
        seen.append(cfg)
        return FakeRetriever.from_config(cfg)

    monkeypatch.setattr(retriever.Retriever, "from_config", staticmethod(spy_from_config))
    cfg = load_config(config)
    tools = InProcessTools.from_config(cfg)

    assert [c.retrieval.sources for c in seen] == expected
    # Everything except the source list is passed through unchanged.
    for c in seen:
        assert c.model_dump(exclude={"retrieval": {"sources"}}) == cfg.model_dump(exclude={"retrieval": {"sources"}})
    assert isinstance(tools, Tools)


def test_from_config_end_to_end_on_the_fake(monkeypatch, chunks_200):
    """c1_docs wired through from_config, with FakeRetriever standing in for the real Retriever."""
    monkeypatch.setattr(retriever.Retriever, "from_config", staticmethod(FakeRetriever.from_config))
    monkeypatch.setattr(retriever, "get_chunk", FakeRetriever(chunks_200).get_chunk)
    tools = InProcessTools.from_config(load_config("c1_docs"))

    docs = tools.search_docs("append a row to a DataFrame", "pandas", version="2.2.2", top_k=3)
    assert docs and all(h.chunk.source == "docs" for h in docs)
    assert tools.search_stackoverflow("append a row to a DataFrame", "pandas") == []
    assert tools.get_chunk(docs[0].chunk_id) == docs[0].chunk
