"""Score a generation with the task's hidden tests. Owner: Roberto."""

from __future__ import annotations

from snakepp.schemas import EvalResult, Generation, Task


def score(generation: Generation, task: Task) -> EvalResult:
    raise NotImplementedError("Roberto: scorer + append-only results store (Week 9)")
