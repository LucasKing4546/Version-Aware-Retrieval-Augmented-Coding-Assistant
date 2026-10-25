"""Pipeline entry point: config -> retrieve -> generate -> evaluate. Owner: Kevin.

`run` handles one config and one benchmark; `sweep` loops over configs and benchmarks and skips pairs that
already have results. LLM calls and sandbox runs are cached, so interrupted runs resume almost for free.
Results go to data/results/<benchmark>/<config>_<date>[_n].{jsonl,json} and are never overwritten.
"""

from __future__ import annotations

import collections
import glob
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import typer
import yaml
from tqdm import tqdm

from snakepp import cache
from snakepp.config import Config, load_config
from snakepp.evaluation.metrics import pass_at_1
from snakepp.evaluation.registry import Benchmark, load_benchmark
from snakepp.generation import llm
from snakepp.generation.citations import extract_citations, extract_code
from snakepp.generation.prompts import SYSTEM, build_prompt
from snakepp.schemas import EvalResult, Generation, Retrieved, Task

RESULTS_DIR = Path("data/results")
QUERY_CHARS = 2000  # retrieval query = start of the task prompt

app = typer.Typer(help="Snake++ experiment runner")


def make_retriever(cfg: Config):
    if not cfg.retrieval.enabled:
        return None
    from snakepp.retrieval import Retriever

    return Retriever.from_disk(cfg)


def retrieve_for(task: Task, retriever) -> list[Retrieved]:
    return retriever.retrieve(task.prompt[:QUERY_CHARS], task.target_versions) if retriever else []


def generate_one(task: Task, cfg: Config, retriever, hits: list[Retrieved] | None = None) -> Generation:
    hits = retrieve_for(task, retriever) if hits is None else hits
    chunks = [retriever.chunks[h.chunk_id] for h in hits] if retriever else []
    prompt = build_prompt(task, chunks, cfg.generation)
    out = llm.complete(prompt, cfg.generation, SYSTEM)
    cited = [chunks[n - 1].id for n in extract_citations(out["text"]) if 1 <= n <= len(chunks)]
    return Generation(
        task_id=task.task_id, config=cfg.name, prompt_hash=cache.key(prompt)[:16], code=extract_code(out["text"]),
        cited_chunk_ids=cited, retrieved=hits, raw_output=out["text"], latency_s=out["latency_s"],
    )


def summarize(cfg: Config, benchmark: str, gens: list[Generation], results: list[EvalResult]) -> dict:
    errors = collections.Counter(r.error_type for r in results if not r.passed)
    return {
        "config": cfg.name, "benchmark": benchmark, "n": len(results), "pass@1": round(pass_at_1(results), 4),
        "errors": dict(errors.most_common()),
        "mean_latency_s": round(statistics.fmean(g.latency_s for g in gens), 3) if gens else 0.0,
        "cited_share": round(sum(bool(g.cited_chunk_ids) for g in gens) / len(gens), 4) if gens else 0.0,
        "model": cfg.generation.model, "date": date.today().isoformat(),
    }


def result_stem(benchmark: str, config: str, root: Path = RESULTS_DIR) -> Path:
    """data/results/<benchmark>/<config>_<date>, with _2, _3 ... when that name is taken."""
    base = root / benchmark / f"{config}_{date.today().isoformat()}"
    stem, n = base, 1
    while stem.with_suffix(".json").exists():
        n += 1
        stem = base.with_name(f"{base.name}_{n}")
    return stem


def write_results(stem: Path, gens: list[Generation], results: list[EvalResult], summary: dict) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    by_id = {r.task_id: r for r in results}
    with open(stem.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps({"generation": g.model_dump(), "result": by_id[g.task_id].model_dump()}) + "\n")
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_benchmark(
    cfg: Config, bench: Benchmark, name: str, workers: int = 4, gen_workers: int = 1
) -> tuple[list, list, dict]:
    """Retrieval runs sequentially; LLM calls run `gen_workers` at a time (vLLM batches concurrent requests)."""
    retriever = make_retriever(cfg)
    hits = [retrieve_for(t, retriever) for t in tqdm(bench.tasks, desc=f"retrieve {cfg.name}", unit="task",
                                                       disable=retriever is None)]
    with ThreadPoolExecutor(max_workers=gen_workers) as pool:
        jobs = pool.map(lambda th: generate_one(th[0], cfg, retriever, th[1]), zip(bench.tasks, hits, strict=True))
        gens = list(tqdm(jobs, total=len(bench.tasks), desc=f"{cfg.name}/{name}", unit="task"))
    results = bench.evaluate({g.task_id: g.code for g in gens}, cfg.name, workers=workers)
    return gens, results, summarize(cfg, name, gens, results)


def apply_sets(cfg: Config, sets: list[str]) -> Config:
    """'retrieval.top_k=5' style overrides for tuning; the config name records them (c5_full+top_k=5)."""
    if not sets:
        return cfg
    data = cfg.model_dump()
    for item in sets:
        dotted, value = item.split("=", 1)
        *parents, leaf = dotted.split(".")
        target = data
        for key in parents:
            target = target[key]
        if leaf not in target:
            raise typer.BadParameter(f"unknown config field {dotted!r}")
        target[leaf] = yaml.safe_load(value)
    data["name"] = cfg.name + "".join(f"+{s.split('.')[-1]}" for s in sets)
    return Config.model_validate(data)


def _with_overrides(cfg: Config, model: str | None, backend: str | None, sets: list[str] | None = None) -> Config:
    gen = cfg.generation.model_copy(update={k: v for k, v in {"model": model, "backend": backend}.items() if v})
    return apply_sets(cfg.model_copy(update={"generation": gen}), sets or [])


@app.command()
def run(
    config: str, benchmark: str = "ds1000", limit: int | None = None, model: str | None = None,
    backend: str | None = None, workers: int = 4, gen_workers: int = 1, set: list[str] | None = None,  # noqa: A002
) -> None:
    """Run one configuration on one benchmark. --model/--backend override the config (e.g. for LM Studio);
    --set retrieval.top_k=5 overrides any field (tune only on ds1000_dev / mbpp_dev)."""
    cfg = _with_overrides(load_config(config), model, backend, set)
    gens, results, summary = run_benchmark(cfg, load_benchmark(benchmark, limit), benchmark, workers, gen_workers)
    stem = result_stem(benchmark, cfg.name)
    write_results(stem, gens, results, summary)
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"saved {stem}.jsonl")


@app.command()
def sweep(
    configs: str = "configs/*.yaml", benchmarks: list[str] | None = None, limit: int | None = None,
    model: str | None = None, backend: str | None = None, force: bool = False, gen_workers: int = 1,
) -> None:
    """Run every config on every benchmark, skipping pairs that already have a results file (unless --force)."""
    for path in sorted(glob.glob(configs)):
        cfg = _with_overrides(load_config(path), model, backend)
        for bench in benchmarks or ["ds1000"]:
            if not force and list((RESULTS_DIR / bench).glob(f"{cfg.name}_[0-9]*.json")):
                typer.echo(f"skip {cfg.name} on {bench}: results exist")
                continue
            try:
                run(path, bench, limit, model, backend, gen_workers=gen_workers)
            except FileNotFoundError as e:  # e.g. Stack Overflow corpus not built yet
                typer.echo(f"skip {cfg.name} on {bench}: {e}")


if __name__ == "__main__":
    app()
