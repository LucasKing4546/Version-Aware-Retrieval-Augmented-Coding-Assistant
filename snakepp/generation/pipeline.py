"""Fixed retrieve-then-generate pipeline. Owner: Enea."""

from __future__ import annotations

from snakepp.config import Config
from snakepp.generation.tools import Tools
from snakepp.schemas import Generation, Task


def answer(task: Task, cfg: Config, tools: Tools) -> Generation:
    raise NotImplementedError("Enea: answer() on FakeRetriever (Week 8)")
