"""Decontamination — owner: Lucas.

Finds Stack Overflow chunks that near-duplicate a benchmark prompt: MinHash LSH over 5-word shingles proposes
candidates, the exact shingle Jaccard (and optionally the embedding cosine) confirms them. Writes
data/decontam/matches.csv, so_clean.parquet (matches removed) and so_leaky.parquet (everything, for c5_leaky).
Usage: python -m snakepp.corpus.decontam run --benchmarks ds1000 --jaccard 0.3
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import typer
from datasketch import MinHash, MinHashLSH

from snakepp.corpus.io import CHUNKS_DIR, read_chunks, write_chunks
from snakepp.schemas import Chunk, Task

OUT_DIR = Path("data/decontam")
DEFAULT_BENCHMARKS = ["ds1000", "humaneval", "mbpp"]
_WORD = re.compile(r"[a-z0-9_]+")

app = typer.Typer(help="Remove benchmark leaks from the Stack Overflow corpus")


@dataclass(frozen=True)
class Match:
    task_id: str
    chunk_id: str
    jaccard: float
    cosine: float | None = None


def shingles(text: str, n: int = 5) -> set[str]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def minhash(grams: set[str], num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for g in grams:
        m.update(g.encode("utf-8"))
    return m


def find_matches(tasks: list[Task], chunks: list[Chunk], jaccard: float = 0.3, num_perm: int = 128) -> list[Match]:
    """Pairs whose shingle Jaccard is >= jaccard. LSH runs at a lower threshold so it misses few true pairs."""
    lsh = MinHashLSH(threshold=max(0.05, jaccard * 0.7), num_perm=num_perm)
    grams = {}
    for c in chunks:
        grams[c.id] = shingles(c.text)
        if grams[c.id]:
            lsh.insert(c.id, minhash(grams[c.id], num_perm))
    matches = []
    for t in tasks:
        tg = shingles(t.prompt)
        if not tg:
            continue
        for cid in lsh.query(minhash(tg, num_perm)):
            j = len(tg & grams[cid]) / len(tg | grams[cid])
            if j >= jaccard:
                matches.append(Match(t.task_id, cid, round(j, 4)))
    return sorted(matches, key=lambda m: (-m.jaccard, m.task_id, m.chunk_id))


def confirm_with_cosine(matches: list[Match], tasks: dict[str, Task], chunks: dict[str, Chunk], encoder,
                        threshold: float) -> list[Match]:
    """Keep matches whose embedding cosine is >= threshold (encoder returns normalized vectors)."""
    if not matches:
        return []
    tv = encoder([tasks[m.task_id].prompt for m in matches])
    cv = encoder([chunks[m.chunk_id].text for m in matches])
    out = []
    for m, a, b in zip(matches, tv, cv, strict=True):
        cos = float((a * b).sum())
        if cos >= threshold:
            out.append(Match(m.task_id, m.chunk_id, m.jaccard, round(cos, 4)))
    return out


def write_outputs(matches: list[Match], chunks: list[Chunk], chunks_dir: Path = CHUNKS_DIR,
                  out_dir: Path = OUT_DIR) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "matches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "chunk_id", "jaccard", "cosine"])
        w.writerows([m.task_id, m.chunk_id, m.jaccard, m.cosine] for m in matches)
    leaked = {m.chunk_id for m in matches}
    clean = write_chunks([c for c in chunks if c.id not in leaked], chunks_dir / "so_clean.parquet")
    leaky = write_chunks(chunks, chunks_dir / "so_leaky.parquet")
    return clean, leaky


@app.command()
def run(
    benchmarks: list[str] | None = None, jaccard: float = 0.3,
    cosine: float | None = None, so_all: Path = CHUNKS_DIR / "so_all.parquet",
) -> None:
    """Match benchmark prompts against so_all.parquet; --cosine also requires an embedding-cosine check."""
    from snakepp.evaluation.registry import load_benchmark

    chunks = read_chunks(so_all)
    tasks = [t for b in benchmarks or DEFAULT_BENCHMARKS for t in load_benchmark(b).tasks]
    matches = find_matches(tasks, chunks, jaccard)
    if cosine is not None:
        from snakepp.config import RetrievalConfig
        from snakepp.retrieval.dense import sentence_transformer_encoder

        encoder = sentence_transformer_encoder(RetrievalConfig().dense_model)
        matches = confirm_with_cosine(matches, {t.task_id: t for t in tasks}, {c.id: c for c in chunks}, encoder,
                                      cosine)
    clean, leaky = write_outputs(matches, chunks)
    typer.echo(f"{len(matches)} matches over {len({m.task_id for m in matches})} tasks; "
               f"so_clean={clean} so_leaky={leaky}")


if __name__ == "__main__":
    app()
