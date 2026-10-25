"""BM25 over chunks with an identifier-aware tokenizer — owner: Kevin.

Identifiers are kept whole (`dataframe.append`) and also split on '.', '_', parentheses and camelCase
(`dataframe`, `append`), so both exact API names and plain words match.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from snakepp.schemas import Chunk, Retrieved

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i in is it my of on or the this to with what when".split()
)


def _parts(identifier: str) -> list[str]:
    parts = []
    for piece in re.split(r"[._]+", identifier):
        parts.extend(_CAMEL.findall(piece))
    return [p.lower() for p in parts]


def tokenize(text: str) -> list[str]:
    """Lowercased tokens: whole compound identifiers plus their parts, minus stopwords."""
    tokens: list[str] = []
    for ident in _IDENT.findall(text):
        parts = _parts(ident)
        if len(parts) > 1:
            tokens.append(ident.lower())
        tokens.extend(p for p in parts if p not in STOPWORDS)
    return tokens


class BM25Index:
    """BM25Okapi over chunk texts (title included, it usually names the API)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        self.ids = [c.id for c in chunks]
        self._bm25 = BM25Okapi([tokenize(f"{c.title}\n{c.text}") for c in chunks], k1=k1, b=b)

    def search(self, query: str, k: int = 50) -> list[Retrieved]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.ids[i]))
        hits = [i for i in order[:k] if scores[i] > 0]
        return [
            Retrieved(chunk_id=self.ids[i], rank=r, score=float(scores[i]), stage="bm25")
            for r, i in enumerate(hits, start=1)
        ]
