"""Benchmark loaders. Reference solutions are dropped at load time. Owner: Roberto."""

from __future__ import annotations

from snakepp.schemas import Benchmark, Split, Task


def load_tasks(benchmark: Benchmark, split: Split) -> list[Task]:
    raise NotImplementedError("Roberto: mini benchmark (Wed 18 Nov), full loaders (Fri 20 Nov)")
