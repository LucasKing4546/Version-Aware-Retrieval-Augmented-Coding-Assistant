"""Parse code and citations from model output — owner: Enea."""

from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_SOURCES = re.compile(r"Sources?:\s*((?:\[\d+\][,\s]*)+)", re.IGNORECASE)


def extract_code(output: str) -> str:
    """First fenced code block; falls back to the whole output when there is no fence."""
    m = _FENCE.search(output)
    return (m.group(1) if m else output).strip()


def extract_citations(output: str) -> list[int]:
    """Numbers from a 'Sources: [1], [3]' line, in order, without duplicates."""
    m = _SOURCES.search(output)
    if not m:
        return []
    seen: list[int] = []
    for n in re.findall(r"\[(\d+)\]", m.group(1)):
        if int(n) not in seen:
            seen.append(int(n))
    return seen
