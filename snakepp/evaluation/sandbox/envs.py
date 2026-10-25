"""Small per-version Python environments under data/envs, created with uv (pip install uv)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def env_python(env: Path) -> Path:
    return env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def ensure_env(env: Path, requirements: list[str], python: str = "3.10") -> Path:
    """Interpreter of `env`, creating it with `requirements` installed when it does not exist yet."""
    exe = env_python(env)
    if not exe.exists():
        uv = [sys.executable, "-m", "uv"]
        subprocess.run([*uv, "venv", "-q", "--python", python, str(env)], check=True)
        subprocess.run([*uv, "pip", "install", "-q", "--python", str(env), *requirements], check=True)
    return exe
