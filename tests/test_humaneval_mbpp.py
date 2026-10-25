import gzip
import json
import sys

import pytest

from snakepp import cache
from snakepp.evaluation.humaneval import HumanEval, function_program
from snakepp.evaluation.mbpp import MBPP

PROMPT = 'from typing import List\n\n\ndef add(xs: List[int]) -> int:\n    """Sum."""\n'


@pytest.fixture(autouse=True)
def _sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SNAKEPP_SANDBOX_PYTHON", sys.executable)
    monkeypatch.setenv("SNAKEPP_SANDBOX", "local")


def _humaneval(tmp_path):
    row = {"task_id": "HumanEval/0", "prompt": PROMPT, "canonical_solution": "SECRET", "entry_point": "add",
           "test": "def check(f):\n    assert f([1, 2]) == 3\n"}
    path = tmp_path / "he.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return HumanEval(path)


def test_function_program_both_styles():
    assert function_program(PROMPT, "    return sum(xs)\n", "add") == PROMPT + "    return sum(xs)\n"
    full = "def add(xs):\n    return sum(xs)\n"
    assert function_program(PROMPT, full, "add") == "from typing import List\n" + full


def test_humaneval_evaluate(tmp_path):
    bench = _humaneval(tmp_path)
    assert "SECRET" not in bench.tasks[0].model_dump_json()
    ok = bench.evaluate({"HumanEval/0": "def add(xs):\n    return sum(xs)"}, "c0")[0]
    bad = bench.evaluate({"HumanEval/0": "    return 0\n"}, "c0")[0]
    assert ok.passed and not bad.passed and bad.error_type == "AssertionError"


def test_mbpp_splits_and_evaluate(tmp_path):
    rows = [{"task_id": i, "prompt": f"Write f{i}.", "code": "SECRET", "test_imports": [],
             "test_list": [f"assert f{i}(1) == 2"]} for i in (5, 11, 520)]
    path = tmp_path / "mbpp.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    test, dev = MBPP(path), MBPP(path, split="dev")
    assert [t.task_id for t in test.tasks] == ["mbpp/11"] and [t.task_id for t in dev.tasks] == ["mbpp/520"]
    assert "assert f11(1) == 2" in test.tasks[0].prompt and "SECRET" not in test.tasks[0].prompt
    assert test.evaluate({"mbpp/11": "def f11(x):\n    return x + 1"}, "c0")[0].passed
