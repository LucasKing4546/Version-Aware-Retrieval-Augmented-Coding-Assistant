"""OverlapScorer: a deterministic stand-in for the cross-encoder, so rerank tests need no model.

Score = number of distinct query tokens in the passage, divided by the passage's token count (so a short, focused
passage beats a long one that merely mentions the words). Counts the pairs it scored, for the cache tests.
"""

from __future__ import annotations

from snakepp.retrieval.tokenizer import tokenize


class OverlapScorer:
    def __init__(self) -> None:
        self.scored = 0

    def __call__(self, query: str, passages: list[str]) -> list[float]:
        self.scored += len(passages)
        words = set(tokenize(query))
        out = []
        for text in passages:
            tokens = tokenize(text)
            out.append(len(words & set(tokens)) / (len(tokens) or 1))
        return out
