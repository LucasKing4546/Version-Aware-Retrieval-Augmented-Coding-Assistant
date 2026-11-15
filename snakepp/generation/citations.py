"""Reading the model's reply: the code block and the cited context entries. Owner: Enea."""

from __future__ import annotations

import re

from snakepp.schemas import Retrieved

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)
_OPEN_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n")
_CITATION = re.compile(r"\[(\d+)\]")
_PYTHON_TAGS = {"python", "py", "python3", ""}


def extract_code(reply: str) -> str:
    """The first ```python (or untagged) fenced block; the whole reply if there is no fence.

    A reply cut off by max_tokens inside its code block has no closing fence: the code after the
    opening fence is returned.
    """
    for tag, body in _FENCE.findall(reply):
        if tag.lower() in _PYTHON_TAGS:
            return body.strip("\n")
    if (opened := _OPEN_FENCE.search(reply)) and opened.group(1).lower() in _PYTHON_TAGS:
        return reply[opened.end() :].strip("\n")
    return reply.strip()


def parse_citations(reply: str, hits: list[Retrieved]) -> list[str]:
    """Chunk ids for the [n] markers outside code blocks, in order of first mention.

    Markers inside code are ignored (`df.iloc[1]` is not a citation). Numbers outside 1..len(hits)
    point at nothing and are dropped.
    """
    prose = _FENCE.sub(" ", reply)
    ids: list[str] = []
    for number in _CITATION.findall(prose):
        n = int(number)
        if 1 <= n <= len(hits) and hits[n - 1].chunk_id not in ids:
            ids.append(hits[n - 1].chunk_id)
    return ids
