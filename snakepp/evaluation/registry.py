"""Benchmark registry: name -> object with `.tasks: list[Task]` and `.evaluate(codes, config) -> list[EvalResult]`."""

from __future__ import annotations

from typing import Protocol

from snakepp.schemas import EvalResult, Task


class Benchmark(Protocol):
    tasks: list[Task]

    def evaluate(self, codes: dict[str, str], config: str, workers: int = 4) -> list[EvalResult]: ...


def load_benchmark(name: str, limit: int | None = None) -> Benchmark:
    if name == "ds1000":
        from snakepp.evaluation.ds1000 import DS1000

        return DS1000(limit=limit)
    if name == "humaneval":
        from snakepp.evaluation.humaneval import HumanEval

        return HumanEval(limit=limit)
    if name == "mbpp":
        from snakepp.evaluation.mbpp import MBPP

        return MBPP(limit=limit)
    if name == "version_set":
        from snakepp.benchmarks.version_set.loader import VersionSet

        return VersionSet(limit=limit)
    raise ValueError(f"unknown benchmark {name!r}: use ds1000, humaneval, mbpp or version_set")
