"""Agent loop v1 (ARCHITECTURE.md §6.2). Owner: Enea.

    Plan -> [DetectVersion] -> Search <-> Judge -> Answer -> [Run -> Repair -> (Search) -> Answer]* -> Done

The code owns the control flow; the model decides at three points (Plan, Judge, Answer). Plan and
Judge reply in JSON; a reply that does not parse falls back to a safe default (Plan: search like the
fixed pipeline; Judge: the evidence is sufficient), so a weak controller degrades to the pipeline.

Bounds: at most MAX_SEARCHES search rounds, MAX_RUNS executions and MAX_CONTEXT chunks in the prompt;
hitting a cap forces Answer. The agent only uses `Tools`, so it has no access an MCP client lacks, and
`run_python` gets the task's sample input, never its hidden tests.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from snakepp.config import Config
from snakepp.generation import llm
from snakepp.generation.citations import extract_code, parse_citations
from snakepp.generation.pipeline import DEFAULT_LIBRARY, corpus_hash, merge
from snakepp.generation.prompts import build_judge_messages, build_messages, build_plan_messages
from snakepp.generation.tools import Tools
from snakepp.retrieval.versions import ALIASES
from snakepp.schemas import Generation, Retrieved, RunResult, Source, Task

MAX_SEARCHES = 3  # search rounds; one round queries every enabled source
MAX_RUNS = 2
MAX_CONTEXT = 10  # chunks in the answer prompt, newest round first

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class Plan:
    needs_retrieval: bool
    library: str | None
    sources: list[Source]
    query: str


@dataclass(frozen=True)
class Verdict:
    sufficient: bool
    query: str | None


def solve(task: Task, cfg: Config, tools: Tools) -> Generation:
    return _Run(task, cfg, tools).solve()


class _Run:
    def __init__(self, task: Task, cfg: Config, tools: Tools) -> None:
        self.task, self.cfg, self.tools = task, cfg, tools
        self.enabled: list[Source] = [] if cfg.retrieval.mode == "none" else list(cfg.retrieval.sources)
        self.hits: list[Retrieved] = []
        self.searches = 0
        self.runs = 0
        self.tool_calls: list[dict[str, Any]] = []

    def solve(self) -> Generation:
        start = time.perf_counter()
        plan = self.plan()
        self.library = self.task.library or (ALIASES.get(plan.library, plan.library) if plan.library else None)
        self.library = self.library or DEFAULT_LIBRARY
        self.versions = self.target_versions()
        self.prompt_task = self.task.model_copy(update={"target_versions": self.versions})
        sources = [s for s in self.enabled if s in plan.sources] or self.enabled

        if plan.needs_retrieval and sources:
            query: str | None = plan.query
            while query and self.searches < MAX_SEARCHES:
                if not self.search(query, sources):
                    break
                query = None if self.searches >= MAX_SEARCHES else self.judge().query

        messages, raw = self.answer()
        while self.task.sample is not None and self.runs < MAX_RUNS:
            code = extract_code(raw)
            result = self.run(code)
            if result is None or result.ok:
                break
            if self.enabled and self.searches < MAX_SEARCHES:
                self.search(_error_query(result), self.enabled)
            messages, raw = self.answer(repair=(code, result))

        return Generation(
            task_id=self.task.task_id,
            config=self.cfg.name,
            prompt_hash=llm.cache_key(messages, self.cfg.generation),
            code=extract_code(raw),
            cited_chunk_ids=parse_citations(raw, self.hits),
            retrieved=[h.model_copy(update={"chunk": None}) for h in self.hits],
            tool_calls=self.tool_calls,
            raw_output=raw,
            latency_s=round(time.perf_counter() - start, 3),
            corpus_hash=corpus_hash(),
        )

    # ---- the model's decisions -------------------------------------------------------------------

    def plan(self) -> Plan:
        data = parse_json(llm.complete(build_plan_messages(self.task), self.cfg.generation)) or {}
        sources = [s for s in data.get("sources") or [] if s in ("docs", "stackoverflow")]
        library = data.get("library")
        query = data.get("query")
        return Plan(
            needs_retrieval=data.get("needs_retrieval") is not False,  # only an explicit false skips retrieval
            library=library.strip().lower() if isinstance(library, str) and library.strip() else None,
            sources=sources or list(self.enabled),
            query=query.strip() if isinstance(query, str) and query.strip() else self.task.prompt,
        )

    def judge(self) -> Verdict:
        reply = llm.complete(build_judge_messages(self.prompt_task, self.hits), self.cfg.generation)
        data = parse_json(reply) or {}
        query = data.get("query")
        if data.get("sufficient") is not False or not isinstance(query, str) or not query.strip():
            return Verdict(sufficient=True, query=None)
        return Verdict(sufficient=False, query=query.strip())

    def answer(self, repair: tuple[str, RunResult] | None = None) -> tuple[list[dict[str, str]], str]:
        messages = build_messages(self.prompt_task, self.hits, repair=repair)
        return messages, llm.complete(messages, self.cfg.generation)

    # ---- tools -----------------------------------------------------------------------------------

    def target_versions(self) -> dict[str, str]:
        if self.task.target_versions:
            return dict(self.task.target_versions)
        found = self._call("detect_versions", {"text_chars": len(self.task.prompt)}, self.tools.detect_versions)(
            self.task.prompt
        )
        return found or {}

    def search(self, query: str, sources: list[Source]) -> list[Retrieved]:
        """One round: every source once, merged to k, added in front of what is already in context."""
        self.searches += 1
        version = self.versions.get(self.library)
        k = self.cfg.retrieval.k
        tool = {"docs": self.tools.search_docs, "stackoverflow": self.tools.search_stackoverflow}
        per_source = []
        for source in sources:
            name = "search_docs" if source == "docs" else "search_stackoverflow"
            args = {"query": query, "library": self.library, "version": version, "top_k": k}
            per_source.append(self._call(name, args, tool[source])(query, self.library, version, k) or [])
        new = merge(per_source, k)
        seen = {h.chunk_id for h in new}
        combined = new + [h for h in self.hits if h.chunk_id not in seen]
        self.hits = [h.model_copy(update={"rank": i}) for i, h in enumerate(combined[:MAX_CONTEXT], start=1)]
        return new

    def run(self, code: str) -> RunResult | None:
        """None when the sandbox itself failed: then there is nothing to repair."""
        self.runs += 1
        args = {"code_chars": len(code), "target_versions": self.versions, "has_sample": True}
        return self._call("run_python", args, self.tools.run_python)(code, self.versions, self.task.sample)

    def _call(self, tool: str, args: dict[str, Any], fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a tool call: log {"tool", "args", "latency_s"} (+ "error") into Generation.tool_calls.

        A failing tool is logged and returns None, so one bad call does not end the task. A tool that
        is not built yet (NotImplementedError) still raises: results without it would be meaningless.
        """

        def call(*a: Any) -> Any:
            start = time.perf_counter()
            entry: dict[str, Any] = {"tool": tool, "args": args}
            try:
                return fn(*a)
            except NotImplementedError:
                raise
            except Exception as exc:  # logged; the agent carries on
                entry["error"] = f"{type(exc).__name__}: {exc}"
                return None
            finally:
                entry["latency_s"] = round(time.perf_counter() - start, 3)
                self.tool_calls.append(entry)

        return call


def parse_json(reply: str) -> dict[str, Any] | None:
    """The first-to-last brace span of a reply, without <think> blocks or ```json fences."""
    text = _THINK.sub("", reply)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _error_query(result: RunResult) -> str:
    """The last non-empty stderr line ("AttributeError: 'DataFrame' object has no attribute 'append'")."""
    lines = [ln.strip() for ln in result.stderr_tail.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return "timeout" if result.timed_out else str(result.error_type)
