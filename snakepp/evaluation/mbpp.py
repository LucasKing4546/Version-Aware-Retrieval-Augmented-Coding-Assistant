"""MBPP (sanitized) harness — owner: Enea.

google-research/mbpp sanitized-mbpp.json; reference code dropped at load. Splits follow the original paper:
test = task ids 11-510 (reported), dev = 511-600 (the only split we tune on). The first test assert is shown in
the prompt, as in Austin et al. (2021), so the model knows the function name and signature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from snakepp.evaluation.humaneval import PYTHON_TARGET, download, score
from snakepp.schemas import EvalResult, Task

DATA_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"
DATA_PATH = Path("data/raw/mbpp/sanitized-mbpp.json")
SPLITS = {"test": range(11, 511), "dev": range(511, 601)}


class MBPP:
    def __init__(self, path: Path = DATA_PATH, limit: int | None = None,
                 split: Literal["test", "dev"] = "test") -> None:
        self.tasks: list[Task] = []
        self._rows: dict[str, dict] = {}
        for row in json.loads(download(DATA_URL, path).read_text(encoding="utf-8")):
            row.pop("code", None)  # never keep reference solutions
            if row["task_id"] not in SPLITS[split]:
                continue
            task_id = f"mbpp/{row['task_id']}"
            self._rows[task_id] = row
            prompt = f"{row['prompt']}\nYour code should pass this test:\n{row['test_list'][0]}"
            self.tasks.append(Task(task_id=task_id, benchmark="mbpp", library="python", prompt=prompt,
                                   target_versions=PYTHON_TARGET))
        self.tasks = self.tasks[:limit] if limit is not None else self.tasks

    def program(self, task_id: str, code: str) -> str:
        row = self._rows[task_id]
        tests = "\n".join([*row.get("test_imports", []), *row["test_list"]])
        return f"{code}\n\n{tests}\n"

    def evaluate(self, codes: dict[str, str], config: str, workers: int = 4) -> list[EvalResult]:
        ids = list(codes)
        return score("mbpp", config, ids, [self.program(t, codes[t]) for t in ids], workers)
