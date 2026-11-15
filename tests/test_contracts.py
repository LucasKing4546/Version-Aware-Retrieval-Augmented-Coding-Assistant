"""The contract: types, configs, layout and stub signatures agreed on Mon 16 Nov."""

import inspect
from datetime import date

import pytest
import yaml

from snakepp import paths
from snakepp.config import CONFIGS_DIR, Config, load_config, load_pins
from snakepp.evaluation import sandbox, scorer, tasks
from snakepp.generation import agent, llm, pipeline
from snakepp.generation.tools import Tools
from snakepp.ingest import chunks
from snakepp.retrieval import retriever, versions
from snakepp.schemas import G1Item, G2Item, G3Item, GoldChunkRef, Retrieved, RunResult, Task


def test_run_result_ok():
    assert RunResult(stdout="1\n").ok
    assert not RunResult(error_type="AttributeError").ok
    assert not RunResult(timed_out=True).ok


def test_retrieved_carries_chunk(chunks_200):
    r = Retrieved(chunk_id=chunks_200[0].id, rank=1, score=0.9, stage="reranked", chunk=chunks_200[0])
    assert Retrieved.model_validate_json(r.model_dump_json()) == r
    with pytest.raises(ValueError):
        Retrieved(chunk_id="x", rank=0, score=0.0, stage="bm25")


def test_task_has_no_solution_field():
    assert not {"solution", "reference", "tests", "canonical_solution"} & set(Task.model_fields)


def test_gold_ref_matches_by_url_and_heading(chunks_200):
    c = chunks_200[0].model_copy(update={"heading_path": "pandas.concat"})
    assert GoldChunkRef(url=c.url).matches(c)
    assert GoldChunkRef(url=c.url, heading_path="pandas.concat").matches(c)
    assert not GoldChunkRef(url=c.url, heading_path="other").matches(c)


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("c*.yaml")), ids=lambda p: p.stem)
def test_configs_load(path):
    cfg = load_config(path)
    assert cfg.name == path.stem


def test_config_keys():
    assert load_config("c0_none").retrieval.mode == "none"
    full = load_config("c5_full")
    assert (full.retrieval.mode, full.retrieval.rerank, full.version_filter.mode) == ("hybrid", True, "off")
    assert load_config("c5_hard").version_filter.mode == "hard"
    assert load_config("c5_soft_01").version_filter.penalty == 0.1
    assert load_config("c5_leaky").retrieval.corpus == "leaky"
    assert Config(name="x").retrieval.k == 5


def test_pins():
    envs = load_pins().environments
    assert envs["pandas_old"]["pandas"].startswith("1.") and envs["pandas_new"]["pandas"].startswith("2.")
    assert all("python" in env for env in envs.values())


def test_gold_files_parse():
    assert [G1Item.model_validate(x) for x in yaml.safe_load(paths.G1.read_text(encoding="utf-8"))] == []
    assert [G3Item.model_validate(x) for x in yaml.safe_load(paths.G3.read_text(encoding="utf-8"))] == []
    lines = [ln for ln in paths.G2.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert all(G2Item.model_validate_json(ln) for ln in lines)


def test_layout():
    assert paths.chunk_path("docs", "pandas", "2.2.2").relative_to(paths.DATA).as_posix() == (
        "chunks/docs/pandas/2.2.2.parquet"
    )
    assert paths.chunk_path("stackoverflow", "pandas", None).name == "unversioned.parquet"
    p = paths.results_path("ds1000", "c5_full", date(2026, 11, 22))
    assert p.relative_to(paths.DATA).as_posix() == "results/ds1000/c5_full_2026-11-22.jsonl"


STUBS = [
    (chunks.load_chunks, ["paths"]),
    (retriever.Retriever.from_config, ["cfg"]),
    (retriever.Retriever.retrieve, ["self", "query", "library", "target_versions", "k"]),
    (retriever.get_chunk, ["chunk_id"]),
    (versions.detect_versions, ["text"]),
    (versions.list_indexed_versions, ["library"]),
    (llm.complete, ["messages", "cfg"]),
    (pipeline.answer, ["task", "cfg", "tools"]),
    (agent.solve, ["task", "cfg", "tools"]),
    (tasks.load_tasks, ["benchmark", "split"]),
    (sandbox.run, ["code", "target_versions", "test", "sample", "timeout_s"]),
    (scorer.score, ["generation", "task"]),
]


@pytest.mark.parametrize("fn,params", STUBS, ids=lambda x: getattr(x, "__qualname__", ""))
def test_signatures(fn, params):
    assert list(inspect.signature(fn).parameters) == params


def test_sandbox_defaults():
    sig = inspect.signature(sandbox.run)
    assert (sig.parameters["test"].default, sig.parameters["sample"].default, sig.parameters["timeout_s"].default) == (
        None,
        None,
        30,
    )


def test_tools_protocol_has_the_six_mcp_tools():
    names = {n for n in dir(Tools) if not n.startswith("_")}
    assert names == {
        "search_docs",
        "search_stackoverflow",
        "detect_versions",
        "list_indexed_versions",
        "get_chunk",
        "run_python",
    }
