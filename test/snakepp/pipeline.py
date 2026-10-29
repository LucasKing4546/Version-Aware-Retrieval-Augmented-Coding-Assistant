"""Pipeline entry point: config -> retrieve -> generate -> evaluate. Owner: Kevin.

TODO (Kevin, M2 Sun 1 Nov): wire snakepp.retrieval.retrieve, snakepp.generation and
snakepp.evaluation together. `run` handles one config and one benchmark; `sweep` loops over
configs and benchmarks, is resumable, and skips work already in data/cache or data/results.
"""

from __future__ import annotations

import glob

import typer

from snakepp.config import load_config

app = typer.Typer(help="Snake++ experiment runner")


@app.command()
def run(config: str, benchmark: str = "ds1000", limit: int | None = None) -> None:
    """Run one configuration on one benchmark."""
    cfg = load_config(config)
    typer.echo(f"[stub] would run {cfg.name} on {benchmark} (limit={limit})")
    raise typer.Exit(code=0)


@app.command()
def sweep(configs: str = "configs/*.yaml", benchmarks: list[str] | None = None) -> None:
    """Run every config on every benchmark."""
    for path in sorted(glob.glob(configs)):
        cfg = load_config(path)
        for bench in benchmarks or ["ds1000"]:
            typer.echo(f"[stub] would run {cfg.name} on {bench}")


if __name__ == "__main__":
    app()
