"""agent.solve on the fakes: FakeRetriever behind InProcessTools, scripted FakeLLM, scripted runner."""

from __future__ import annotations

import json

import pytest

from snakepp import paths
from snakepp.config import load_config
from snakepp.generation import agent, llm
from snakepp.generation.agent import MAX_CONTEXT, parse_json
from snakepp.generation.inprocess_tools import InProcessTools
from snakepp.generation.prompts import SYSTEM_JUDGE, SYSTEM_NO_CONTEXT, SYSTEM_PLAN, SYSTEM_WITH_CONTEXT
from snakepp.schemas import Chunk, RunResult, Task
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.fake_retriever import FakeRetriever

TASK = Task(
    task_id="g1-pandas-append",
    benchmark="g1",
    prompt="Append a row to a DataFrame.",
    library="pandas",
    target_versions={"pandas": "2.2.2"},
)
APPEND_ERROR = RunResult(
    error_type="AttributeError",
    stderr_tail="Traceback (most recent call last):\nAttributeError: 'DataFrame' object has no attribute 'append'\n",
)


def plan(needs: bool = True, query: str = "concat dataframe rows", sources=("docs", "stackoverflow"), library=None):
    return json.dumps({"needs_retrieval": needs, "library": library, "sources": list(sources), "query": query})


def judge(sufficient: bool = True, query: str | None = None) -> str:
    return json.dumps({"sufficient": sufficient, "query": query})


def code(body: str = "df = pd.concat([df, row])", sources: str = "Sources: [1]") -> str:
    return f"```python\n{body}\n```\n{sources}"


class ScriptedTools(InProcessTools):
    """InProcessTools over FakeRetriever, with a scripted sandbox and controllable versions."""

    def __init__(self, chunks: list[Chunk], runs: list[RunResult] | None = None, detected=None) -> None:
        self.run_results = list(runs or [])
        self.ran: list[tuple[str, dict, str | None]] = []
        super().__init__(
            retrievers={s: FakeRetriever(chunks, sources=[s]) for s in ("docs", "stackoverflow")},
            get_chunk=FakeRetriever(chunks).get_chunk,
            detect_versions=lambda text: dict(detected or {}),
            list_indexed_versions=lambda library: {},
            run=self._run,
        )

    def _run(self, code: str, target_versions: dict[str, str], *, sample: str | None = None) -> RunResult:
        self.ran.append((code, target_versions, sample))
        return self.run_results.pop(0) if self.run_results else RunResult(stdout="ok\n")


@pytest.fixture(autouse=True)
def no_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MANIFEST", tmp_path / "manifest.json")


def steps(fake: FakeLLM) -> list[str]:
    names = {SYSTEM_PLAN: "plan", SYSTEM_JUDGE: "judge", SYSTEM_WITH_CONTEXT: "answer", SYSTEM_NO_CONTEXT: "answer"}
    return [names[call[0]["content"]] for call in fake.calls]


def tools_used(gen) -> list[str]:
    return [c["tool"] for c in gen.tool_calls]


# ---- control flow ------------------------------------------------------------------------------


def test_no_retrieval_needed(monkeypatch, chunks_200):
    fake = FakeLLM(script=[plan(needs=False), code("print(sum([1, 2]))", "")]).install(monkeypatch)
    gen = agent.solve(TASK, load_config("c5_full"), ScriptedTools(chunks_200))
    assert steps(fake) == ["plan", "answer"] and gen.tool_calls == [] and gen.retrieved == []
    assert gen.code == "print(sum([1, 2]))" and fake.remaining == 0


def test_one_search_then_answer(monkeypatch, chunks_200):
    fake = FakeLLM(script=[plan(), judge(True), code()]).install(monkeypatch)
    cfg = load_config("c5_full")
    gen = agent.solve(TASK, cfg, ScriptedTools(chunks_200))

    assert steps(fake) == ["plan", "judge", "answer"]
    assert tools_used(gen) == ["search_docs", "search_stackoverflow"]
    for call in gen.tool_calls:
        assert call["args"] == {"query": "concat dataframe rows", "library": "pandas", "version": "2.2.2", "top_k": 5}
        assert call["latency_s"] >= 0 and "error" not in call
    assert len(gen.retrieved) == cfg.retrieval.k and all(r.chunk is None for r in gen.retrieved)
    assert gen.cited_chunk_ids == [gen.retrieved[0].chunk_id]
    assert gen.prompt_hash == llm.cache_key(fake.calls[-1], cfg.generation)


def test_judge_rewrites_until_the_search_cap(monkeypatch, chunks_200):
    script = [plan(sources=["docs"]), judge(False, "pandas concat"), judge(False, "DataFrame append removed"), code()]
    fake = FakeLLM(script=script).install(monkeypatch)
    gen = agent.solve(TASK, load_config("c5_full"), ScriptedTools(chunks_200))
    # Three rounds, judged twice: after the third search the cap forces Answer without a judge call.
    assert steps(fake) == ["plan", "judge", "judge", "answer"]
    assert [c["args"]["query"] for c in gen.tool_calls] == [
        "concat dataframe rows",
        "pandas concat",
        "DataFrame append removed",
    ]
    assert tools_used(gen) == ["search_docs"] * 3  # the plan picked docs only
    assert len(gen.retrieved) <= MAX_CONTEXT


def test_run_and_repair(monkeypatch, chunks_200):
    task = TASK.model_copy(update={"sample": "df = pd.DataFrame({'a': [1]})"})
    tools = ScriptedTools(chunks_200, runs=[APPEND_ERROR, RunResult(stdout="ok")])
    fake = FakeLLM(
        script=[plan(), judge(True), code("df = df.append(row)"), code("df = pd.concat([df, row])")]
    ).install(monkeypatch)
    gen = agent.solve(task, load_config("c5_full"), tools)

    assert steps(fake) == ["plan", "judge", "answer", "answer"]
    assert gen.code == "df = pd.concat([df, row])"
    assert [r[0] for r in tools.ran] == ["df = df.append(row)", "df = pd.concat([df, row])"]
    assert all(r[1:] == ({"pandas": "2.2.2"}, task.sample) for r in tools.ran)  # sample only, never tests
    # The error text became a repair search, and the repair prompt shows the failed code and the error.
    repair_queries = [c["args"]["query"] for c in gen.tool_calls if c["tool"].startswith("search")][2:]
    assert repair_queries == ["AttributeError: 'DataFrame' object has no attribute 'append'"] * 2
    repair_prompt = fake.calls[-1][1]["content"]
    assert "df = df.append(row)" in repair_prompt and "it raised AttributeError" in repair_prompt
    assert tools_used(gen).count("run_python") == 2


def test_run_cap_forces_a_final_answer(monkeypatch, chunks_200):
    task = TASK.model_copy(update={"sample": "x"})
    tools = ScriptedTools(chunks_200, runs=[APPEND_ERROR, APPEND_ERROR])
    script = [plan(needs=False), code("a"), code("b"), code("c")]
    fake = FakeLLM(script=script).install(monkeypatch)
    gen = agent.solve(task, load_config("c5_full"), tools)
    assert steps(fake) == ["plan", "answer", "answer", "answer"]
    assert len(tools.ran) == 2 and gen.code == "c"


def test_no_sample_means_no_run(monkeypatch, chunks_200):
    FakeLLM(script=[plan(needs=False), code()]).install(monkeypatch)
    tools = ScriptedTools(chunks_200)
    agent.solve(TASK, load_config("c5_full"), tools)
    assert tools.ran == []


# ---- configs and fallbacks ---------------------------------------------------------------------


def test_c0_none_never_searches(monkeypatch, chunks_200):
    task = TASK.model_copy(update={"sample": "x"})
    FakeLLM(script=[plan(), code("a"), code("b")]).install(monkeypatch)
    gen = agent.solve(task, load_config("c0_none"), ScriptedTools(chunks_200, runs=[APPEND_ERROR]))
    assert tools_used(gen) == ["run_python", "run_python"]  # no search, not even for the repair


def test_plan_sources_outside_the_config_fall_back_to_it(monkeypatch, chunks_200):
    FakeLLM(script=[plan(sources=["stackoverflow"]), judge(True), code()]).install(monkeypatch)
    gen = agent.solve(TASK, load_config("c1_docs"), ScriptedTools(chunks_200))
    assert tools_used(gen) == ["search_docs"]


def test_unparseable_plan_searches_like_the_pipeline(monkeypatch, chunks_200):
    FakeLLM(script=["I think we should look at the docs!", judge(True), code()]).install(monkeypatch)
    gen = agent.solve(TASK, load_config("c5_full"), ScriptedTools(chunks_200))
    assert tools_used(gen) == ["search_docs", "search_stackoverflow"]
    assert gen.tool_calls[0]["args"]["query"] == TASK.prompt


def test_unparseable_judge_counts_as_sufficient(monkeypatch, chunks_200):
    fake = FakeLLM(script=[plan(), "hmm", code()]).install(monkeypatch)
    agent.solve(TASK, load_config("c5_full"), ScriptedTools(chunks_200))
    assert steps(fake) == ["plan", "judge", "answer"]


def test_versions_are_detected_when_the_task_has_none(monkeypatch, chunks_200):
    task = Task(task_id="q1", benchmark="mini", prompt="With pandas==1.5.3, append a row.", library="pandas")
    FakeLLM(script=[plan(), judge(True), code()]).install(monkeypatch)
    gen = agent.solve(task, load_config("c1_docs"), ScriptedTools(chunks_200, detected={"pandas": "1.5.3"}))
    assert tools_used(gen) == ["detect_versions", "search_docs"]
    assert gen.tool_calls[1]["args"]["version"] == "1.5.3"


def test_plan_library_alias_is_resolved(monkeypatch, chunks_200):
    task = Task(task_id="q2", benchmark="mini", prompt="Fit a linear regression.", target_versions={"x": "1"})
    FakeLLM(script=[plan(library="Sklearn"), judge(True), code()]).install(monkeypatch)
    gen = agent.solve(task, load_config("c1_docs"), ScriptedTools(chunks_200))
    assert gen.tool_calls[0]["args"]["library"] == "scikit-learn"


def test_failing_tool_is_logged_and_the_agent_carries_on(monkeypatch, chunks_200):
    tools = ScriptedTools(chunks_200)

    def broken(*args, **kwargs):
        raise RuntimeError("index offline")

    monkeypatch.setattr(tools, "search_docs", broken)
    FakeLLM(script=[plan(sources=["docs"]), code()]).install(monkeypatch)
    gen = agent.solve(TASK, load_config("c1_docs"), tools)
    assert gen.tool_calls[0]["error"] == "RuntimeError: index offline"
    assert gen.retrieved == [] and gen.code == "df = pd.concat([df, row])"


def test_unbuilt_tool_still_raises(monkeypatch, chunks_200):
    tools = ScriptedTools(chunks_200)

    def unbuilt(*args, **kwargs):
        raise NotImplementedError("Kevin")

    monkeypatch.setattr(tools, "search_docs", unbuilt)
    FakeLLM(script=[plan(sources=["docs"])]).install(monkeypatch)
    with pytest.raises(NotImplementedError):
        agent.solve(TASK, load_config("c1_docs"), tools)


# ---- parse_json --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('<think>maybe {"a": 2}</think>\n{"a": 1}', {"a": 1}),
        ('Sure! {"a": {"b": 2}} Hope that helps.', {"a": {"b": 2}}),
        ("no json here", None),
        ("{broken", None),
        ("[1, 2]", None),
    ],
)
def test_parse_json(reply, expected):
    assert parse_json(reply) == expected
