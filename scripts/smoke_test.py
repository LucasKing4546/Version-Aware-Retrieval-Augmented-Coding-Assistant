"""End-to-end smoke test of everything built so far. Prints PASS / FAIL / SKIP per check.

Usage: python scripts/smoke_test.py [--github] [--fast]
Run from the repo root inside the venv. Works on any branch: checks for modules that are not on the current
branch yet are SKIPPED, not failed. --github also checks the issue board and CI (needs `gh` logged in);
--fast skips the full pytest and ruff run. Exit code is 1 if any check fails.
"""

from __future__ import annotations

import collections
import http.server
import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "chunks_200.jsonl"


class Skip(Exception):
    """Raised by a check whose code is not on this branch yet."""


def _need(path: str) -> Path:
    p = ROOT / path
    if not p.exists():
        raise Skip(f"{path} not on this branch")
    return p


def _need_attr(module: str, name: str):
    mod = importlib.import_module(module)
    if not hasattr(mod, name):
        raise Skip(f"{module}.{name} not on this branch")
    return getattr(mod, name)


def _load_script(name: str):
    path = _need(f"scripts/{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(*cmd: str) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError((proc.stdout + proc.stderr).strip()[-800:])
    return proc.stdout.strip()


def _load_chunks():
    from snakepp.schemas import Chunk

    return [Chunk.model_validate_json(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------- checks: each returns a one-line detail


def check_pytest() -> str:
    return _run(sys.executable, "-m", "pytest", "-q").splitlines()[-1]


def check_ruff() -> str:
    return _run(sys.executable, "-m", "ruff", "check", ".")


def check_configs() -> str:
    from snakepp.config import load_config

    names = sorted(load_config(p).name for p in (ROOT / "configs").glob("*.yaml"))
    assert {"c0_none", "c5_full", "c5_full_vf"} <= set(names), names
    assert load_config(ROOT / "configs" / "c0_none.yaml").retrieval.enabled is False
    return f"{len(names)} configs load: {', '.join(names)}"


def check_fixture_contents() -> str:
    chunks = _load_chunks()
    sources = collections.Counter(c.source for c in chunks)
    assert len(chunks) == 200 and len({c.id for c in chunks}) == 200
    assert sources == {"docs": 130, "stackoverflow": 70}, sources
    return f"200 unique chunks, {dict(sources)}, {len({c.library for c in chunks})} libraries"


def check_fixture_deterministic() -> str:
    assert _load_script("make_chunk_fixture").build_chunks() == _load_chunks(), "regenerate the fixture"
    return "generator output == committed file"


def check_version_filter() -> str:
    from snakepp.retrieval.version_filter import apply, compatible
    from snakepp.schemas import Retrieved

    chunks = _load_chunks()
    target = {"pandas": "2.2.2"}
    bad = [c for c in chunks if not compatible(c, target)]
    assert bad and all(c.library == "pandas" and c.version.startswith("1.") for c in bad)
    results = [Retrieved(chunk_id=c.id, rank=i + 1, score=1.0, stage="bm25") for i, c in enumerate(chunks[:60])]
    by_id = {c.id: c for c in chunks}
    hard = apply(results, by_id, target, "hard")
    assert all(compatible(by_id[r.chunk_id], target) for r in hard)
    return f"{len(bad)} pandas 1.x chunks incompatible with pandas 2.2.2; hard mode keeps {len(hard)}/60"


def check_ds1000_versions() -> str:
    pinned = _need_attr("snakepp.evaluation.ds1000", "pinned_versions")()
    expected = {"pandas": "1.5.3", "numpy": "1.26.4", "matplotlib": "3.8.4", "scikit-learn": "1.4.0", "scipy": "1.12.0"}
    assert expected.items() <= pinned.items(), pinned
    return ", ".join(f"{k} {v}" for k, v in expected.items())


def check_watermark_filter() -> str:
    is_watermark = _need_attr("snakepp.corpus.stackoverflow", "is_watermark")
    assert is_watermark({"Id": "1000000001", "OwnerUserId": "-1"})
    assert is_watermark({"Id": "1000000010"})
    assert not is_watermark({"Id": "11227809", "OwnerUserId": "-1"})
    return "fake v1 rows dropped, real posts kept"


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        start = int(self.headers.get("Range", "bytes=0-")[6:].split("-")[0] or 0)
        body = self.payload[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


def check_download_resume() -> str:
    script = _load_script("download_so_dump")
    payload = bytes(range(256)) * 4096  # 1 MiB
    _RangeHandler.payload = payload
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dump.7z"
            out.write_bytes(payload[:300_000])  # an interrupted download
            script.download(f"http://127.0.0.1:{server.server_port}/dump.7z", out)
            assert out.read_bytes() == payload, "resumed file differs"
    finally:
        server.shutdown()
    return "interrupted 1 MiB download resumed from byte 300000 and matches"


def check_related_work() -> str:
    text = _need("report/related_work.md").read_text(encoding="utf-8")
    sections = [line for line in text.splitlines() if line.startswith("## ")]
    verified = text.count("✔")
    assert len(sections) >= 7 and verified >= 4
    return f"{len(sections)} sections, {verified} verified references"


def check_github() -> str:
    issues = json.loads(_run("gh", "issue", "list", "--state", "all", "--limit", "500", "--json", "number"))
    milestones = json.loads(_run("gh", "api", "repos/{owner}/{repo}/milestones?state=all"))
    runs = json.loads(_run("gh", "run", "list", "--branch", "main", "--limit", "1", "--json", "conclusion"))
    assert len(issues) >= 39 and len(milestones) == 6, (len(issues), len(milestones))
    assert runs and runs[0]["conclusion"] == "success", runs
    return f"{len(issues)} issues, {len(milestones)} milestones, last CI on main: success"


# ---------------------------------------------------------------- runner


def run_checks(checks: list[tuple[str, Callable[[], str]]]) -> dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for name, check in checks:
        try:
            status, detail = "PASS", check()
        except Skip as e:
            status, detail = "SKIP", str(e)
        except Exception as e:  # report every failure, keep going
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        counts[status] += 1
        first_line = (detail.splitlines() or [""])[0]
        typer.echo(f"[{status}] {name:<28} {first_line}")
    return counts


def main(github: bool = False, fast: bool = False) -> None:
    sys.path.insert(0, str(ROOT))
    checks = [
        ("configs load", check_configs),
        ("fixture contents", check_fixture_contents),
        ("fixture deterministic", check_fixture_deterministic),
        ("version filter on fixture", check_version_filter),
        ("DS-1000 pinned versions", check_ds1000_versions),
        ("SO watermark filter", check_watermark_filter),
        ("SO download resume", check_download_resume),
        ("related-work outline", check_related_work),
    ]
    if not fast:
        checks = [("pytest", check_pytest), ("ruff", check_ruff), *checks]
    if github:
        checks.append(("GitHub board + CI", check_github))
    counts = run_checks(checks)
    typer.echo(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
    raise typer.Exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    typer.run(main)
