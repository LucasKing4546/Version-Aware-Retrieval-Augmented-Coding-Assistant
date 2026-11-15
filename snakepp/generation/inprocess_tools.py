"""InProcessTools: the 6 MCP tools as direct Python calls. Owner: Enea.

Same behaviour the MCP server must expose (ARCHITECTURE.md §7), without the transport. Used by
the fixed pipeline and the agent in tests (wired to fakes), and as the in-process arm of the
MCP parity experiment E9 (wired to the real components by `from_config`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from snakepp.config import Config
from snakepp.schemas import Chunk, Retrieved, RunResult, Source

MAX_FETCH = 8  # with a shared backend, ask for at most this many times top_k hits


class SearchBackend(Protocol):
    """Anything with `Retriever.retrieve`'s signature: the real retriever or a fake."""

    def retrieve(self, query: str, library: str | None, target_versions: dict[str, str], k: int) -> list[Retrieved]: ...


class Runner(Protocol):
    """`snakepp.evaluation.sandbox.run`'s signature, minus the arguments the tools never pass."""

    def __call__(self, code: str, target_versions: dict[str, str], *, sample: str | None = None) -> RunResult: ...


class InProcessTools:
    """Implements `snakepp.generation.tools.Tools` by calling the components directly.

    `retrievers` maps each enabled source to its backend. A source without a backend (for example
    Stack Overflow under `c1_docs`, or both under `c0_none`) returns no hits, so a config's source
    ablation also holds for the agent.
    """

    def __init__(
        self,
        *,
        retrievers: Mapping[Source, SearchBackend],
        get_chunk: Callable[[str], Chunk],
        detect_versions: Callable[[str], dict[str, str]],
        list_indexed_versions: Callable[[str | None], dict[str, list[str]]],
        run: Runner,
    ) -> None:
        self._retrievers = dict(retrievers)
        self._get_chunk = get_chunk
        self._detect_versions = detect_versions
        self._list_indexed_versions = list_indexed_versions
        self._run = run

    @classmethod
    def from_config(cls, cfg: Config) -> InProcessTools:
        """Wire the real components: one retriever per source enabled in `cfg.retrieval`."""
        from snakepp.evaluation import sandbox
        from snakepp.retrieval import retriever, versions

        retrievers: dict[Source, SearchBackend] = {}
        if cfg.retrieval.mode != "none":
            for source in cfg.retrieval.sources:
                retrievers[source] = retriever.Retriever.from_config(_single_source(cfg, source))
        return cls(
            retrievers=retrievers,
            get_chunk=retriever.get_chunk,
            detect_versions=versions.detect_versions,
            list_indexed_versions=versions.list_indexed_versions,
            run=sandbox.run,
        )

    # ---- the 6 tools ---------------------------------------------------------------------------

    def search_docs(self, query: str, library: str, version: str | None = None, top_k: int = 5) -> list[Retrieved]:
        return self._search("docs", query, library, version, top_k)

    def search_stackoverflow(
        self, query: str, library: str, version: str | None = None, top_k: int = 5
    ) -> list[Retrieved]:
        return self._search("stackoverflow", query, library, version, top_k)

    def detect_versions(self, text: str) -> dict[str, str]:
        return self._detect_versions(text)

    def list_indexed_versions(self, library: str | None = None) -> dict[str, list[str]]:
        return self._list_indexed_versions(_normalise(library) if library else None)

    def get_chunk(self, chunk_id: str) -> Chunk:
        """Full chunk with attribution. Raises KeyError for an unknown id."""
        return self._get_chunk(chunk_id)

    def run_python(self, code: str, target_versions: dict[str, str], sample: str | None = None) -> RunResult:
        # Only the sample input is forwarded: hidden benchmark tests never reach the agent (PLAN.md §5).
        return self._run(code, target_versions, sample=sample)

    # ---- helpers -------------------------------------------------------------------------------

    def _search(self, source: Source, query: str, library: str, version: str | None, top_k: int) -> list[Retrieved]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        backend = self._retrievers.get(source)
        if backend is None:
            return []
        library = _normalise(library)
        target_versions = {library: version} if version else {}
        # A backend shared by both sources can fill top_k with the other source's hits, so ask again
        # for more until top_k survive the filter, the backend runs out, or MAX_FETCH x top_k is reached.
        # A per-source backend (what from_config builds) is called exactly once.
        want = top_k
        while True:
            hits = backend.retrieve(query, library, target_versions, want)
            out = self._keep_source(hits, source, top_k)
            if len(out) == top_k or len(hits) < want or want >= MAX_FETCH * top_k:
                return out
            want = min(want * 2, MAX_FETCH * top_k)

    def _keep_source(self, hits: list[Retrieved], source: Source, top_k: int) -> list[Retrieved]:
        """Every hit carries its chunk (the MCP output includes text and attribution), only this
        tool's source survives, and ranks are renumbered from 1 after filtering."""
        out: list[Retrieved] = []
        for hit in sorted(hits, key=lambda h: h.rank):
            chunk = hit.chunk or self._get_chunk(hit.chunk_id)
            if chunk.source != source:
                continue
            out.append(hit.model_copy(update={"rank": len(out) + 1, "chunk": chunk}))
            if len(out) == top_k:
                break
        return out


def _normalise(library: str) -> str:
    return library.strip().lower()


def _single_source(cfg: Config, source: Source) -> Config:
    return cfg.model_copy(update={"retrieval": cfg.retrieval.model_copy(update={"sources": [source]})})
