"""Create the local sandbox interpreter: Python 3.10 + DS-1000's pinned versions, in data/envs/ds1000.

Usage: python scripts/make_sandbox_env.py [--env data/envs/ds1000]
Uses `uv` (pip install uv) to fetch Python 3.10 when it is not installed. Use this where Docker is not available;
on Linux with Docker, build snakepp/evaluation/sandbox/Dockerfile instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

VERSIONS = Path("snakepp/evaluation/sandbox/versions.txt")


def commands(env: Path, python: str = "3.10") -> list[list[str]]:
    uv = [sys.executable, "-m", "uv"]
    return [
        [*uv, "python", "install", python],
        [*uv, "venv", "--python", python, str(env)],
        [*uv, "pip", "install", "--python", str(env), "-r", str(VERSIONS)],
    ]


def main(env: Path = Path("data/envs/ds1000")) -> None:
    for cmd in commands(env):
        typer.echo(" ".join(cmd))
        subprocess.run(cmd, check=True)
    typer.echo(f"sandbox env ready: {env}")


if __name__ == "__main__":
    typer.run(main)
