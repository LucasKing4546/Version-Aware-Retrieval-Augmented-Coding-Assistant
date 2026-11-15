"""The MCP server end to end through a real MCP client (in-process transport), on the real fixture with BM25."""

from pathlib import Path

import anyio
import pytest
from mcp import Client

from snakepp.config import Config, RetrievalConfig, VersionFilterConfig
from snakepp.ingest.chunks import load_chunks
from snakepp.schemas import Chunk, Retrieved
from snakepp.server.mcp_server import build_server, tools_from_chunks

FIXTURE = Path(__file__).parent / "fixtures" / "fixture_chunks.jsonl"


@pytest.fixture(scope="module")
def real_chunks() -> list[Chunk]:
    return list(load_chunks(FIXTURE))


def server_for(chunks, mode="bm25", vf="off"):
    cfg = Config(
        name="t", retrieval=RetrievalConfig(mode=mode, rerank=False), version_filter=VersionFilterConfig(mode=vf)
    )
    return build_server(tools_from_chunks(cfg, chunks))


def call(server, tool: str, **args):
    """One tool call through an MCP client session; returns the CallToolResult."""

    async def go():
        async with Client(server) as client:
            return await client.call_tool(tool, args)

    return anyio.run(go)


def hits(result) -> list[Retrieved]:
    assert not result.is_error, result.content[0].text
    return [Retrieved.model_validate(h) for h in result.structured_content["result"]]


def test_lists_the_week9_tools_as_read_only(real_chunks):
    async def go():
        async with Client(server_for(real_chunks)) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in anyio.run(go)}
    assert set(tools) == {"search_docs", "search_stackoverflow", "get_chunk"}
    assert all(t.annotations.read_only_hint for t in tools.values())
    assert set(tools["search_docs"].input_schema["properties"]) == {"query", "library", "version", "top_k"}
    assert tools["search_docs"].input_schema["required"] == ["query", "library"]


def test_search_docs(real_chunks):
    out = hits(call(server_for(real_chunks), "search_docs", query="DataFrame.append", library="pandas", top_k=3))
    assert [h.rank for h in out] == [1, 2, 3]
    assert all(h.chunk.source == "docs" and h.chunk.library == "pandas" for h in out)
    assert "DataFrame.append" in out[0].chunk.url


def test_version_reaches_the_filter(real_chunks):
    server = server_for(real_chunks, vf="hard")
    out = hits(call(server, "search_docs", query="DataFrame.append iteritems", library="pandas", version="2.2.3"))
    assert out and all(h.chunk.version == "2.2.3" for h in out)


def test_search_stackoverflow_keeps_attribution(real_chunks):
    out = hits(call(server_for(real_chunks), "search_stackoverflow", query="append row", library="Pandas"))
    assert out and all(h.chunk.source == "stackoverflow" and h.chunk.author and h.chunk.license for h in out)
    assert all(h.chunk.version_inferred for h in out)


def test_get_chunk_and_unknown_id(real_chunks):
    server = server_for(real_chunks)
    result = call(server, "get_chunk", chunk_id=real_chunks[0].id)
    assert Chunk.model_validate(result.structured_content) == real_chunks[0]
    missing = call(server, "get_chunk", chunk_id="nope")
    assert missing.is_error and "unknown chunk id 'nope'" in missing.content[0].text


def test_bad_arguments_come_back_as_tool_errors(real_chunks):
    server = server_for(real_chunks)
    zero = call(server, "search_docs", query="append", library="pandas", top_k=0)
    assert zero.is_error and "top_k must be >= 1" in zero.content[0].text
    assert call(server, "search_docs", query="append").is_error  # library is required


def test_no_retrieval_config_returns_nothing(real_chunks):
    assert hits(call(server_for(real_chunks, mode="none"), "search_docs", query="append", library="pandas")) == []
