"""DS-1000 harness — owner: Enea.

Loads the official release (xlang-ai/DS-1000, data/ds1000.jsonl.gz), keeps Pandas, Numpy, Matplotlib, Sklearn
and Scipy, and builds Task objects. Reference solutions are dropped at load time and never stored or shown.
Evaluation follows the official script: code_context + `test_execution(code)` (+ `test_string(code)`).
Splits: `dev` is a fixed 10% of each library's problem groups (a problem and all its perturbations stay together),
used for tuning only; `test` is the rest and is what we report; `all` is everything.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Literal

from snakepp.evaluation.sandbox.runner import run_many
from snakepp.schemas import EvalResult, Task

VERSIONS_TXT = Path(__file__).parent / "sandbox" / "versions.txt"
DATA_URL = "https://github.com/xlang-ai/DS-1000/raw/main/data/ds1000.jsonl.gz"
DATA_PATH = Path("data/raw/ds1000/ds1000.jsonl.gz")

# DS-1000 metadata["library"] -> our Chunk.library names
LIBRARIES = {
    "Pandas": "pandas",
    "Numpy": "numpy",
    "Matplotlib": "matplotlib",
    "Sklearn": "scikit-learn",
    "Scipy": "scipy",
}
INSTRUCTION = (
    "Write only the missing code that goes after BEGIN SOLUTION, so that the program above runs correctly. "
    "Do not repeat the given code."
)
DEV_FRACTION = 0.10
SPLIT_SEED = "snakepp-ds1000-dev-v1"  # changing it changes the dev split: never do so after tuning has started
_MARKERS = ("<code>", "</code>", "BEGIN SOLUTION", "END SOLUTION", "```python", "```")


def pinned_versions(path: Path = VERSIONS_TXT) -> dict[str, str]:
    """Read `name==version` lines from the sandbox requirements, skipping comments and blank lines."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def target_versions(ds1000_library: str) -> dict[str, str]:
    """Task.target_versions for a DS-1000 problem: its own library at the version DS-1000 pins."""
    library = LIBRARIES[ds1000_library]
    return {library: pinned_versions()[library]}


def download(path: Path = DATA_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def clean_code(code: str) -> str:
    """Strip DS-1000 / markdown markers the model may echo around its answer."""
    for marker in _MARKERS:
        code = code.replace(marker, "")
    return code.strip("\n")


def dev_groups(metas: list[dict], fraction: float = DEV_FRACTION) -> set[tuple[str, int]]:
    """(library, origin id) groups in the dev split: the first `fraction` of each library's groups by a seeded hash."""
    groups: dict[str, set[int]] = defaultdict(set)
    for m in metas:
        groups[m["library"]].add(m["perturbation_origin_id"])
    dev = set()
    for library, ids in groups.items():
        ranked = sorted(ids, key=lambda i: hashlib.sha1(f"{SPLIT_SEED}:{library}:{i}".encode()).hexdigest())
        dev |= {(library, i) for i in ranked[: math.ceil(fraction * len(ranked))]}
    return dev


class DS1000:
    """Tasks plus the hidden test code needed to score them."""

    def __init__(self, path: Path = DATA_PATH, libraries: list[str] | None = None, limit: int | None = None,
                 split: Literal["test", "dev", "all"] = "test") -> None:
        keep = set(libraries or LIBRARIES)
        self.tasks: list[Task] = []
        self._contexts: dict[str, str] = {}
        with gzip.open(download(path), "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        for row in rows:
            row.pop("reference_code", None)  # never keep reference solutions
        dev = dev_groups([r["metadata"] for r in rows if r["metadata"]["library"] in keep])
        for row in rows:
            meta = row["metadata"]
            in_dev = (meta["library"], meta["perturbation_origin_id"]) in dev
            if meta["library"] not in keep or (split == "dev" and not in_dev) or (split == "test" and in_dev):
                continue
            task_id = f"ds1000/{meta['problem_id']}"
            self._contexts[task_id] = row["code_context"]
            self.tasks.append(Task(
                task_id=task_id, benchmark="ds1000", prompt=f"{row['prompt'].rstrip()}\n\n{INSTRUCTION}",
                library=LIBRARIES[meta["library"]], target_versions=target_versions(meta["library"]),
            ))
        if limit is not None:
            self.tasks = self.tasks[:limit]

    def program(self, task_id: str, code: str) -> str:
        context = self._contexts[task_id]
        program = f"{context}\ncode = {clean_code(code)!r}\ntest_execution(code)\n"
        return program + ("test_string(code)\n" if "test_string(" in context else "")

    def evaluate(self, codes: dict[str, str], config: str, workers: int = 4) -> list[EvalResult]:
        ids = list(codes)
        outcomes = run_many([self.program(t, codes[t]) for t in ids], timeout=120, workers=workers)
        return [
            EvalResult(task_id=t, benchmark="ds1000", config=config, passed=o.passed,
                       error_type=o.error_type, stderr_tail=o.stderr_tail)
            for t, o in zip(ids, outcomes, strict=True)
        ]
