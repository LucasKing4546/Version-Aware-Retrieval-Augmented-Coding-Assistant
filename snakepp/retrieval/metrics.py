"""Retrieval metrics over a G2 gold set: recall@k, MRR and version-correct@k. Owner: Kevin.

    python -m snakepp.retrieval.metrics --config c4_hybrid --gold gold/g2.jsonl --ks 1,3,5,10

Gold chunks are GoldChunkRef(url, heading_path), so a hit is relevant when its url matches and, if the ref names
a heading path, the heading path matches too (several chunks of one section can all be relevant).

- recall@k: share of a query's gold refs matched by at least one of the top-k hits.
- MRR: 1 / rank of the first relevant hit (0 if none was retrieved), averaged over queries.
- version-correct@k: share of the top-k hits that are not a version mismatch for the query's target versions
  (hits with no version, or for a library with no target, count as correct). Divided by k, so when the hard
  filter returns fewer than k hits the missing slots count as wrong; it measures what the user is shown.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from pathlib import Path

import typer

from snakepp import paths
from snakepp.config import load_config
from snakepp.retrieval.retriever import Retriever
from snakepp.retrieval.versions import compatibility
from snakepp.schemas import Chunk, G2Item, GoldChunkRef


def _relevant(chunk: Chunk, gold: Sequence[GoldChunkRef]) -> bool:
    return any(ref.matches(chunk) for ref in gold)


def recall_at_k(hits: Sequence[Chunk], gold: Sequence[GoldChunkRef], k: int) -> float:
    if not gold:
        raise ValueError("a query needs at least one gold chunk")
    return sum(any(ref.matches(c) for c in hits[:k]) for ref in gold) / len(gold)


def reciprocal_rank(hits: Sequence[Chunk], gold: Sequence[GoldChunkRef]) -> float:
    return next((1.0 / rank for rank, c in enumerate(hits, start=1) if _relevant(c, gold)), 0.0)


def version_correct_at_k(hits: Sequence[Chunk], target_versions: dict[str, str], k: int) -> float:
    return sum(compatibility(c, target_versions) != "mismatch" for c in hits[:k]) / k


def evaluate(retriever: Retriever, items: Sequence[G2Item], ks: Sequence[int] = (1, 3, 5, 10)) -> dict[str, float]:
    """Mean of every metric over the gold items, retrieving max(ks) hits per query."""
    rows: dict[str, list[float]] = {}
    for item in items:
        hits = [h.chunk for h in retriever.retrieve(item.query, item.library, item.target_versions, max(ks))]
        row = {"mrr": reciprocal_rank(hits, item.gold_chunks)}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(hits, item.gold_chunks, k)
            row[f"version_correct@{k}"] = version_correct_at_k(hits, item.target_versions, k)
        for name, value in row.items():
            rows.setdefault(name, []).append(value)
    return {name: statistics.fmean(values) for name, values in rows.items()}


def load_g2(path: Path = paths.G2) -> list[G2Item]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [G2Item.model_validate_json(line) for line in lines if line.strip()]


def main(
    config: str = typer.Option(..., help="config name in configs/, e.g. c4_hybrid"),
    gold: str = typer.Option("gold/g2.jsonl", help="G2 jsonl file"),
    ks: str = typer.Option("1,3,5,10", help="comma-separated cutoffs"),
) -> None:
    items = load_g2(Path(gold))
    if not items:
        typer.echo(f"no gold items in {gold}", err=True)
        raise typer.Exit(1)
    scores = evaluate(Retriever.from_config(load_config(config)), items, [int(k) for k in ks.split(",")])
    typer.echo(f"{config} on {gold} ({len(items)} queries)")
    for name, value in scores.items():
        typer.echo(f"  {name:20} {value:.3f}")


if __name__ == "__main__":
    typer.run(main)
