from snakepp.evaluation.ds1000 import LIBRARIES, pinned_versions, target_versions


def test_pinned_versions_cover_our_libraries():
    pins = pinned_versions()
    assert set(LIBRARIES.values()) <= set(pins)
    assert pins["pandas"] == "1.5.3"


def test_pinned_versions_skips_comments(tmp_path):
    path = tmp_path / "versions.txt"
    path.write_text("# header\n\npandas==1.5.3  # inline\nnumpy\n", encoding="utf-8")
    assert pinned_versions(path) == {"pandas": "1.5.3"}


def test_target_versions_maps_ds1000_names():
    assert target_versions("Sklearn") == {"scikit-learn": "1.4.0"}


def _fake_release(tmp_path):
    import gzip
    import json

    context = "def test_execution(code):\n    ns = {}\n    exec(code, ns)\n    assert ns['result'] == 4\n"
    rows = [
        {"prompt": "Compute 2+2 into result.\nBEGIN SOLUTION\n<code>", "code_context": context,
         "reference_code": "SECRET", "metadata": {"library": "Numpy", "problem_id": 7}},
        {"prompt": "torch", "code_context": context, "reference_code": "SECRET",
         "metadata": {"library": "Pytorch", "problem_id": 8}},
    ]
    path = tmp_path / "ds1000.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return path


def test_ds1000_tasks_drop_solutions(tmp_path):
    from snakepp.evaluation.ds1000 import DS1000

    bench = DS1000(_fake_release(tmp_path))
    assert [t.task_id for t in bench.tasks] == ["ds1000/7"]
    task = bench.tasks[0]
    assert task.library == "numpy" and task.target_versions == {"numpy": "1.26.4"}
    assert "SECRET" not in task.model_dump_json() and "SECRET" not in bench.program("ds1000/7", "x")


def test_ds1000_evaluate(tmp_path, monkeypatch):
    import sys

    from snakepp import cache
    from snakepp.evaluation.ds1000 import DS1000, clean_code

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SNAKEPP_SANDBOX_PYTHON", sys.executable)
    monkeypatch.setenv("SNAKEPP_SANDBOX", "local")
    assert clean_code("```python\nresult = 2 + 2\n```\n</code>") == "result = 2 + 2"
    bench = DS1000(_fake_release(tmp_path))
    ok, bad = bench.evaluate({"ds1000/7": "result = 2 + 2"}, "c0"), bench.evaluate({"ds1000/7": "result = 5"}, "c0")
    assert ok[0].passed and ok[0].config == "c0"
    assert not bad[0].passed and bad[0].error_type == "AssertionError"
