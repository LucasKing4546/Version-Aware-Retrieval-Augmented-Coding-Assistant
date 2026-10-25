import sys

from snakepp import cache
from snakepp.benchmarks.version_set import loader

YAML = """\
- id: add
  library: pandas
  question: "Add one."
  setup: |
    x = 1
  test: |
    assert result == 2
  reference:
    "1.5.3": |
      result = x + 1
    "2.2.2": |
      result = 1 + x
  breaks:
    "2.2.2": |
      result = x
"""


def test_tasks_per_version(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(YAML, encoding="utf-8")
    bench = loader.VersionSet(path)
    assert [t.task_id for t in bench.tasks] == ["vs/add@1.5.3", "vs/add@2.2.2"]
    assert bench.tasks[1].target_versions == {"pandas": "2.2.2"}
    assert "x = 1" in bench.tasks[0].prompt and "result = x + 1" not in bench.tasks[0].prompt


def test_evaluate_uses_version_env(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    seen = []
    monkeypatch.setattr(loader, "env_for", lambda lib, v: seen.append(v) or sys.executable)
    path = tmp_path / "q.yaml"
    path.write_text(YAML, encoding="utf-8")
    res = loader.VersionSet(path).evaluate({"vs/add@1.5.3": "result = x + 1", "vs/add@2.2.2": "result = x"}, "c0")
    assert [r.passed for r in res] == [True, False] and seen == ["1.5.3", "2.2.2"]


def test_validation_runs_and_shipped_questions():
    qs = loader.load_questions()
    assert len(qs) >= 15
    runs = loader.validation_runs(qs)
    assert all(kind in ("reference", "breaks") for _, _, kind, _ in runs)
    assert all(q.get("breaks") for q in qs)  # every question is shown to be version-sensitive
