"""Sandbox runner — owner: Enea.

Runs a test program (generated code + benchmark test) in isolation and reports pass/fail with the error type.
Two backends:
- docker: `docker run --network none --memory 2g` on the image built from this folder (Linux, Kaggle, CI);
- local:  a separate interpreter with the pinned versions (data/envs/ds1000, see scripts/make_sandbox_env.py),
          run with -I in a temp dir. No network isolation, so only run benchmark code you trust.
Set SNAKEPP_SANDBOX=docker|local (default local) and optionally SNAKEPP_SANDBOX_PYTHON / SNAKEPP_SANDBOX_IMAGE.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from snakepp import cache

DEFAULT_ENV = Path("data/envs/ds1000")
DEFAULT_IMAGE = "snakepp-sandbox:ds1000"
_EXC = re.compile(r"^(?:[\w.]+\.)?(\w+(?:Error|Exception|Exit|Interrupt|Warning|Failure))\b")


@dataclass(frozen=True)
class Outcome:
    passed: bool
    error_type: str | None = None
    stderr_tail: str | None = None


def sandbox_python() -> str:
    if env := os.environ.get("SNAKEPP_SANDBOX_PYTHON"):
        return env
    exe = DEFAULT_ENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not exe.exists():
        raise RuntimeError(f"{exe} missing: run `python scripts/make_sandbox_env.py` or set SNAKEPP_SANDBOX_PYTHON")
    return str(exe.resolve())


def error_type(stderr: str, returncode: int) -> str | None:
    """Exception class name from the last traceback line, e.g. 'AttributeError'."""
    if returncode == 0:
        return None
    for line in reversed(stderr.strip().splitlines()):
        if m := _EXC.match(line.strip()):
            return m.group(1)
    return f"ExitCode{returncode}"


def _command(program_path: Path, backend: str) -> list[str]:
    if backend == "docker":
        image = os.environ.get("SNAKEPP_SANDBOX_IMAGE", DEFAULT_IMAGE)
        return ["docker", "run", "--rm", "-i", "--network", "none", "--memory", "2g", "--cpus", "1",
                "-e", "MPLBACKEND=Agg", image, "python", "-"]
    return [sandbox_python(), "-I", str(program_path)]


def run_program(program: str, timeout: float = 30.0, backend: str | None = None) -> Outcome:
    backend = backend or os.environ.get("SNAKEPP_SANDBOX", "local")
    with tempfile.TemporaryDirectory(prefix="snakepp_") as tmp:
        path = Path(tmp) / "program.py"
        path.write_text(program, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME")}
        env.update({"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8",
                    "HOME": tmp, "USERPROFILE": tmp, "MPLCONFIGDIR": tmp})  # matplotlib needs a home dir
        try:
            proc = subprocess.run(
                _command(path, backend), input=program if backend == "docker" else None, cwd=tmp, env=env,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return Outcome(False, "Timeout", f"timed out after {timeout:.0f} s")
    if proc.returncode == 0:
        return Outcome(True)
    return Outcome(False, error_type(proc.stderr, proc.returncode), proc.stderr[-800:] or None)


def run_cached(program: str, timeout: float = 30.0, backend: str | None = None) -> Outcome:
    """run_program, cached in data/cache/sandbox by program + timeout + interpreter."""
    backend = backend or os.environ.get("SNAKEPP_SANDBOX", "local")
    where = os.environ.get("SNAKEPP_SANDBOX_IMAGE", DEFAULT_IMAGE) if backend == "docker" else sandbox_python()
    k = cache.key(program, timeout, backend, where)
    return Outcome(**cache.cached_json("sandbox", k, lambda: asdict(run_program(program, timeout, backend))))


def run_many(
    programs: list[str], timeout: float = 30.0, workers: int = 4, backend: str | None = None, use_cache: bool = True
) -> list[Outcome]:
    """Run programs in parallel, results in input order."""
    run = run_cached if use_cache else run_program
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: run(p, timeout, backend), programs))
