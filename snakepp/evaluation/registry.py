"""Benchmark registry: name -> object with `.tasks: list[Task]` and `.evaluate(codes, config) -> list[EvalResult]`."""

from __future__ import annotations

from typing import Protocol

from snakepp.schemas import EvalResult, Task

# *_dev splits are for tuning only; the others are what we report.
BENCHMARKS = ("ds1000", "humaneval", "mbpp", "version_set", "ds1000_dev", "mbpp_dev")


class Benchmark(Protocol):
    tasks: list[Task]

    def evaluate(self, codes: dict[str, str], config: str, workers: int = 4) -> list[EvalResult]: ...


def load_benchmark(name: str, limit: int | None = None) -> Benchmark:
    if name in ("ds1000", "ds1000_dev"):
        from snakepp.evaluation.ds1000 import DS1000

        return DS1000(limit=limit, split="dev" if name == "ds1000_dev" else "test")
    if name == "humaneval":
        from snakepp.evaluation.humaneval import HumanEval

        return HumanEval(limit=limit)
    if name in ("mbpp", "mbpp_dev"):
        from snakepp.evaluation.mbpp import MBPP

        return MBPP(limit=limit, split="dev" if name == "mbpp_dev" else "test")
    if name == "version_set":
        from snakepp.benchmarks.version_set.loader import VersionSet

        return VersionSet(limit=limit)
    raise ValueError(f"unknown benchmark {name!r}: use {', '.join(BENCHMARKS)}")
