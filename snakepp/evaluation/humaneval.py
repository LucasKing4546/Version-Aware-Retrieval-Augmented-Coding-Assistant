"""HumanEval harness — owner: Enea.

Official release (openai/human-eval, data/HumanEval.jsonl.gz), canonical solutions dropped at load. A program is
the prompt's imports + the model's function when it wrote the whole function, else prompt + completion; then the
benchmark's `check(entry_point)` runs in the shared sandbox.
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

from snakepp.evaluation.sandbox.runner import run_many
from snakepp.schemas import EvalResult, Task

DATA_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
DATA_PATH = Path("data/raw/humaneval/HumanEval.jsonl.gz")
PYTHON_TARGET = {"python": "3.10"}  # the sandbox interpreter
INSTRUCTION = "Complete the function above. Write the whole function, including its signature."


def download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def imports_of(source: str) -> str:
    return "\n".join(line for line in source.splitlines() if line.startswith(("import ", "from ")))


def function_program(prompt: str, code: str, entry_point: str) -> str:
    """Code that defines entry_point: the model's own function if it wrote one, else prompt + completion."""
    if f"def {entry_point}" in code:
        return f"{imports_of(prompt)}\n{code}"
    return f"{prompt}{code}"


def score(benchmark: str, config: str, ids: list[str], programs: list[str], workers: int) -> list[EvalResult]:
    outcomes = run_many(programs, timeout=30, workers=workers)
    return [
        EvalResult(task_id=t, benchmark=benchmark, config=config, passed=o.passed, error_type=o.error_type,
                   stderr_tail=o.stderr_tail)
        for t, o in zip(ids, outcomes, strict=True)
    ]


class HumanEval:
    def __init__(self, path: Path = DATA_PATH, limit: int | None = None) -> None:
        self.tasks: list[Task] = []
        self._rows: dict[str, dict] = {}
        with gzip.open(download(DATA_URL, path), "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                row.pop("canonical_solution", None)  # never keep reference solutions
                self._rows[row["task_id"]] = row
                self.tasks.append(Task(task_id=row["task_id"], benchmark="humaneval", library="python",
                                       prompt=f"```python\n{row['prompt']}```\n\n{INSTRUCTION}",
                                       target_versions=PYTHON_TARGET))
        self.tasks = self.tasks[:limit] if limit is not None else self.tasks

    def program(self, task_id: str, code: str) -> str:
        row = self._rows[task_id]
        body = function_program(row["prompt"], code, row["entry_point"])
        return f"{body}\n\n{row['test']}\n\ncheck({row['entry_point']})\n"

    def evaluate(self, codes: dict[str, str], config: str, workers: int = 4) -> list[EvalResult]:
        ids = list(codes)
        return score("humaneval", config, ids, [self.program(t, codes[t]) for t in ids], workers)
