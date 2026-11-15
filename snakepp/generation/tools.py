"""The 6 MCP tools as a Python protocol (ARCHITECTURE.md §7). Owner: Enea.

Implementations: InProcessTools (Enea, over the fakes), the MCP adapter (Enea, Week 10).
The MCP server (Kevin) exposes exactly these methods with the same arguments.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from snakepp.schemas import Chunk, Retrieved, RunResult


@runtime_checkable
class Tools(Protocol):
    def search_docs(self, query: str, library: str, version: str | None = None, top_k: int = 5) -> list[Retrieved]: ...

    def search_stackoverflow(
        self, query: str, library: str, version: str | None = None, top_k: int = 5
    ) -> list[Retrieved]: ...

    def detect_versions(self, text: str) -> dict[str, str]: ...

    def list_indexed_versions(self, library: str | None = None) -> dict[str, list[str]]: ...

    def get_chunk(self, chunk_id: str) -> Chunk: ...

    def run_python(self, code: str, target_versions: dict[str, str], sample: str | None = None) -> RunResult: ...
