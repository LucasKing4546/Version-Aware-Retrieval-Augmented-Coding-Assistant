from datetime import date

from snakepp import cache, pipeline
from snakepp.config import Config
from snakepp.generation import llm
from snakepp.retrieval import Retriever
from snakepp.schemas import EvalResult, Task


class FakeBench:
    tasks = [Task(task_id=f"t{i}", benchmark="version_set", prompt=f"Use pandas.concat {i}",
                  target_versions={"pandas": "2.2.2"}) for i in range(3)]

    def evaluate(self, codes, config, workers=4):
        return [EvalResult(task_id=t, benchmark="fake", config=config, passed="concat" in c) for t, c in codes.items()]


def _fake_llm(monkeypatch, tmp_path, answer):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "call_backend", lambda prompt, system, cfg: answer)


def test_generate_one_maps_citations(monkeypatch, tmp_path, chunks_200):
    _fake_llm(monkeypatch, tmp_path, "```python\ndf = pd.concat([a, b])\n```\nSources: [2], [9]")
    cfg = Config.model_validate({"name": "c", "retrieval": {"dense": False, "rerank": False}})
    retriever = Retriever(cfg, chunks_200)
    g = pipeline.generate_one(FakeBench.tasks[0], cfg, retriever)
    assert g.code == "df = pd.concat([a, b])" and len(g.retrieved) == 3
    assert g.cited_chunk_ids == [g.retrieved[1].chunk_id]  # [9] is out of range and dropped


def test_run_benchmark_without_retrieval(monkeypatch, tmp_path):
    _fake_llm(monkeypatch, tmp_path, "```python\npd.concat(x)\n```")
    cfg = Config.model_validate({"name": "c0", "retrieval": {"enabled": False}})
    gens, results, summary = pipeline.run_benchmark(cfg, FakeBench(), "fake")
    assert summary["pass@1"] == 1.0 and summary["n"] == 3 and gens[0].retrieved == []


def test_results_never_overwritten(tmp_path):
    first = pipeline.result_stem("ds1000", "c0", tmp_path)
    assert first.name == f"c0_{date.today().isoformat()}"
    first.parent.mkdir(parents=True)
    first.with_suffix(".json").write_text("{}")
    assert pipeline.result_stem("ds1000", "c0", tmp_path).name.endswith("_2")


def test_write_results(tmp_path, monkeypatch):
    _fake_llm(monkeypatch, tmp_path, "```python\nx\n```")
    cfg = Config.model_validate({"name": "c0", "retrieval": {"enabled": False}})
    gens, results, summary = pipeline.run_benchmark(cfg, FakeBench(), "fake")
    stem = tmp_path / "res" / "c0_x"
    pipeline.write_results(stem, gens, results, summary)
    assert len(stem.with_suffix(".jsonl").read_text().splitlines()) == 3
    assert '"pass@1": 0.0' in stem.with_suffix(".json").read_text()
