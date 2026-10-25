"""Metrics — owner: Enea (pass@1) and Kevin (retrieval)."""

from __future__ import annotations

from snakepp.schemas import EvalResult


def pass_at_1(results: list[EvalResult]) -> float:
    return sum(r.passed for r in results) / len(results) if results else 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    return len(set(retrieved_ids[:k]) & gold_ids) / len(gold_ids) if gold_ids else 0.0


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            return 1.0 / i
    return 0.0
