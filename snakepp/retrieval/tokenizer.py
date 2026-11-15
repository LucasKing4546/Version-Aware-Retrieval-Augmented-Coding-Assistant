"""Identifier-aware tokenizer for BM25. Owner: Kevin.

Queries name APIs ("DataFrame.append", "is_monotonic", "iteritems"). A plain tokenizer either keeps
"dataframe.append" whole (so "append" alone misses it) or splits it (so the exact name stops being special).
We emit both: the whole dotted name, each dotted part, and its camelCase / snake_case pieces.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|\d+(?:\.\d+)*")
_PIECE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in into is it its of on or so that the "
    "this to use used using was what when which why will with you your".split()
)


def _pieces(part: str) -> list[str]:
    """camelCase and snake_case pieces of one name: "is_monotonic" -> is, monotonic; "DataFrame" -> data, frame."""
    return [p.lower() for chunk in part.split("_") for p in _PIECE.findall(chunk)]


def tokenize(text: str) -> list[str]:
    """Tokens with repeats kept (BM25 counts them). Version numbers like "2.2.3" stay whole."""
    out: list[str] = []
    for match in _TOKEN.finditer(text):
        word = match.group()
        if word[0].isdigit():
            out.append(word)
            continue
        parts = word.split(".")
        if len(parts) > 1:
            out.append(word.lower())
        for part in parts:
            lower = part.lower()
            if lower in STOPWORDS:
                continue
            out.append(lower)
            pieces = _pieces(part)
            if len(pieces) > 1:
                out.extend(p for p in pieces if p not in STOPWORDS)
    return out
