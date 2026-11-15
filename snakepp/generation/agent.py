"""Agent loop: plan, detect versions, search, judge, answer, run, repair. Owner: Enea."""

from __future__ import annotations

from snakepp.config import Config
from snakepp.generation.tools import Tools
from snakepp.schemas import Generation, Task


def solve(task: Task, cfg: Config, tools: Tools) -> Generation:
    raise NotImplementedError("Enea: agent v1 (Week 9)")
