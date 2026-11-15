"""`ingest` CLI. Owner: Lucas.

    python -m snakepp.ingest.cli docs --library pandas --version 2.2.2
    python -m snakepp.ingest.cli so --tags pandas --since 2015-01-01

Writes data/chunks/<source>/<library>/<version>.parquet (see snakepp.paths) and rewrites data/manifest.json.
Idempotent: re-running on the same source produces the same chunk ids.
"""

from __future__ import annotations

import typer

app = typer.Typer(help="Ingest a source into versioned chunk parquet files.")


@app.command()
def docs(library: str = typer.Option(...), version: str = typer.Option(...)) -> None:
    """Parse and chunk one library's docs at one pinned version."""
    raise NotImplementedError("Lucas: docs parser + heading-aware chunker (Week 8)")


@app.command()
def so(tags: str = typer.Option(...), since: str = typer.Option(..., help="YYYY-MM-DD")) -> None:
    """Fetch, chunk and version-tag Stack Overflow Q+A pairs."""
    raise NotImplementedError("Lucas: Stack Overflow parser + version tagger (Week 9)")


if __name__ == "__main__":
    app()
