"""Run code in a per-version environment: no network, timeout, memory cap. Owner: Roberto."""

from __future__ import annotations

from snakepp.schemas import RunResult


def run(
    code: str,
    target_versions: dict[str, str],
    test: str | None = None,
    sample: str | None = None,
    timeout_s: int = 30,
) -> RunResult:
    raise NotImplementedError("Roberto: LocalRunner (Wed 18 Nov), Docker (Fri 27 Nov)")
