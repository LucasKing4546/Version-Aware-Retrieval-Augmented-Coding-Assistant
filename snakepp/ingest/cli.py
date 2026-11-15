"""`ingest` CLI. Owner: Lucas.

    python -m snakepp.ingest.cli docs --library pandas --version 2.2.2
    python -m snakepp.ingest.cli so --tags pandas --since 2015-01-01
    python -m snakepp.ingest.cli pypi --library pandas

Writes data/chunks/<source>/<library>/<version>.parquet (see snakepp.paths) and rewrites data/manifest.json.
Idempotent: re-running on the same source produces the same chunk ids.
"""

from __future__ import annotations

from datetime import date

import typer

from snakepp.ingest import pypi as pypi_source
from snakepp.ingest import so_fetch

app = typer.Typer(help="Ingest a source into versioned chunk parquet files.")


@app.command()
def docs(library: str = typer.Option(...), version: str = typer.Option(...)) -> None:
    """Parse and chunk one library's docs at one pinned version."""
    raise NotImplementedError("Lucas: docs parser + heading-aware chunker (Week 8)")


@app.command()
def so(
    tags: str = typer.Option(..., help="Comma-separated tags, e.g. pandas,numpy"),
    since: str = typer.Option(..., help="YYYY-MM-DD"),
) -> None:
    """Fetch, chunk and version-tag Stack Overflow Q+A pairs. Set SE_API_KEY for the 10k/day quota."""
    for tag in filter(None, (t.strip() for t in tags.split(","))):
        so_fetch.fetch_questions(tag, date.fromisoformat(since))
    # TODO(Lucas, Week 9): parse raw pages into version-tagged chunks.


@app.command()
def pypi(library: str = typer.Option(..., help="PyPI package name, e.g. pandas")) -> None:
    """Download the release-date table used to infer Stack Overflow versions."""
    releases = pypi_source.fetch_releases(library)
    typer.echo(f"{library}: {len(releases)} releases, latest {next(reversed(releases))}")


if __name__ == "__main__":
    app()
