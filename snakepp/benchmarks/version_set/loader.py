"""Version-sensitive question set — owner: Roberto.

questions.yaml holds questions whose correct answer depends on the installed library version. Each question is a
Task per version (task_id 'vs/<id>@<version>', target_versions={library: version}) and is scored in a separate
environment with exactly that version (data/envs/vs/<library>-<version>, created with uv on first use).
`validate` proves every test passes with the reference answer and fails with the 'breaks' answer.
Usage: python -m snakepp.benchmarks.version_set.loader validate
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from snakepp.evaluation.sandbox import envs
from snakepp.evaluation.sandbox.runner import run_many
from snakepp.schemas import EvalResult, Task

QUESTIONS = Path(__file__).with_name("questions.yaml")
ENVS_DIR = Path("data/envs/vs")
EXTRA_REQUIREMENTS = {("pandas", "1.5.3"): ["numpy<2"]}
INSTRUCTION = "Continue the code above. Keep the variable names. Store the answer in `result`."

app = typer.Typer(help="Version-sensitive question set")


@app.callback()
def _main() -> None:
    """Keep subcommand names on the command line."""


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def env_for(library: str, version: str) -> str:
    reqs = [f"{library}=={version}", *EXTRA_REQUIREMENTS.get((library, version), [])]
    return str(envs.ensure_env(ENVS_DIR / f"{library}-{version}", reqs).resolve())


def program(question: dict, code: str) -> str:
    return f"{question['setup']}\n{code}\n\n{question['test']}"


class VersionSet:
    def __init__(self, path: Path = QUESTIONS, limit: int | None = None) -> None:
        self.tasks: list[Task] = []
        self._by_id: dict[str, tuple[dict, str]] = {}
        for q in load_questions(path):
            for version in sorted(q["reference"]):
                task_id = f"vs/{q['id']}@{version}"
                self._by_id[task_id] = (q, version)
                prompt = f"{q['question']}\n\n```python\n{q['setup']}```\n\n{INSTRUCTION}"
                self.tasks.append(Task(task_id=task_id, benchmark="version_set", prompt=prompt,
                                       library=q["library"], target_versions={q["library"]: version}))
        self.tasks = self.tasks[:limit] if limit is not None else self.tasks

    def evaluate(self, codes: dict[str, str], config: str, workers: int = 4) -> list[EvalResult]:
        ids = list(codes)
        pairs = [self._by_id[t] for t in ids]
        pythons = [env_for(q["library"], v) for q, v in pairs]
        outcomes = run_many([program(q, codes[t]) for t, (q, _) in zip(ids, pairs, strict=True)], timeout=60,
                            workers=workers, pythons=pythons)
        return [
            EvalResult(task_id=t, benchmark="version_set", config=config, passed=o.passed,
                       error_type=o.error_type, stderr_tail=o.stderr_tail)
            for t, o in zip(ids, outcomes, strict=True)
        ]


def validation_runs(questions: list[dict]) -> list[tuple[str, str, str, bool]]:
    """(question id, version, kind, expected_pass) for every reference and breaks answer."""
    runs = []
    for q in questions:
        runs += [(q["id"], v, "reference", True) for v in q["reference"]]
        runs += [(q["id"], v, "breaks", False) for v in q.get("breaks", {})]
    return runs


@app.command()
def validate(path: Path = QUESTIONS) -> None:
    """Run every reference answer (must pass) and breaks answer (must fail) in its version's environment."""
    questions = {q["id"]: q for q in load_questions(path)}
    runs = validation_runs(list(questions.values()))
    programs = [program(questions[qid], questions[qid][kind][v]) for qid, v, kind, _ in runs]
    pythons = [env_for(questions[qid]["library"], v) for qid, v, _, _ in runs]
    outcomes = run_many(programs, timeout=60, workers=4, pythons=pythons, use_cache=False)
    bad = 0
    for (qid, v, kind, expected), o in zip(runs, outcomes, strict=True):
        ok = o.passed == expected
        bad += not ok
        typer.echo(f"[{'ok' if ok else 'BAD'}] {qid}@{v} {kind}: {'pass' if o.passed else o.error_type}")
    typer.echo(f"\n{len(questions)} questions, {len(runs) - bad}/{len(runs)} checks as expected")
    raise typer.Exit(1 if bad else 0)


if __name__ == "__main__":
    app()
