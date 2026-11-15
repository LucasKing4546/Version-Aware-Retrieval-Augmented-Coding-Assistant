"""Fixed retrieve-then-generate pipeline. Owner: Enea.

answer(): search each enabled source through the Tools (so the in-process and MCP paths behave the
same), merge into k chunks in total, prompt, call the LLM once, read the code and citations.
"""

from __future__ import annotations

import json
import time

from snakepp import paths
from snakepp.config import Config
from snakepp.generation import llm
from snakepp.generation.citations import extract_code, parse_citations
from snakepp.generation.prompts import build_messages
from snakepp.generation.tools import Tools
from snakepp.schemas import Generation, Retrieved, Task

DEFAULT_LIBRARY = "python"  # tasks with no third-party library (HumanEval, MBPP) search the stdlib docs


def answer(task: Task, cfg: Config, tools: Tools) -> Generation:
    start = time.perf_counter()
    hits = retrieve_context(task, cfg, tools)
    messages = build_messages(task, hits)
    raw = llm.complete(messages, cfg.generation)  # through the module, so FakeLLM can replace it
    return Generation(
        task_id=task.task_id,
        config=cfg.name,
        prompt_hash=llm.cache_key(messages, cfg.generation),
        code=extract_code(raw),
        cited_chunk_ids=parse_citations(raw, hits),
        retrieved=[h.model_copy(update={"chunk": None}) for h in hits],  # ids + scores; text stays in the store
        raw_output=raw,
        latency_s=round(time.perf_counter() - start, 3),
        corpus_hash=corpus_hash(),
    )


def retrieve_context(task: Task, cfg: Config, tools: Tools) -> list[Retrieved]:
    rc = cfg.retrieval
    if rc.mode == "none" or not rc.sources:
        return []
    library = task.library or DEFAULT_LIBRARY
    version = task.target_versions.get(library)
    search = {"docs": tools.search_docs, "stackoverflow": tools.search_stackoverflow}
    per_source = [search[source](task.prompt, library, version, rc.k) for source in rc.sources]
    return merge(per_source, rc.k)


def merge(per_source: list[list[Retrieved]], k: int) -> list[Retrieved]:
    """k hits in total from several sources, ranks renumbered from 1.

    Cross-encoder scores are comparable across sources, so fully reranked lists merge by score.
    BM25, cosine and RRF scores are not, so otherwise the lists are interleaved by rank.
    """
    flat = [h for hits in per_source for h in hits]
    if flat and all(h.stage == "reranked" for h in flat):
        ordered = sorted(flat, key=lambda h: -h.score)  # stable: ties keep source order
    else:
        ordered = [h for rank in range(max(map(len, per_source), default=0)) for h in _at(per_source, rank)]
    return [h.model_copy(update={"rank": i}) for i, h in enumerate(ordered[:k], start=1)]


def _at(per_source: list[list[Retrieved]], rank: int) -> list[Retrieved]:
    return [hits[rank] for hits in per_source if rank < len(hits)]


def corpus_hash() -> str | None:
    """From data/manifest.json once ingest writes it; None before that."""
    try:
        return json.loads(paths.MANIFEST.read_text(encoding="utf-8")).get("corpus_hash")
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return None
