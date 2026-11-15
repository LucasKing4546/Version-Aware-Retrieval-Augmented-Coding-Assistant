"""PyPI release dates, used to infer a Stack Overflow post's library version from its date."""

from __future__ import annotations

import json
import urllib.request
from datetime import date
from pathlib import Path

from packaging.version import InvalidVersion, Version

from snakepp.paths import DATA

PYPI_DIR = DATA / "pypi"
USER_AGENT = "snakepp-ingest"


def release_dates(releases: dict[str, list[dict]]) -> dict[str, str]:
    """PyPI `releases` map -> {version: first upload date (ISO)}, final releases only, oldest version first."""
    out = {}
    for v, files in releases.items():
        try:
            parsed = Version(v)
        except InvalidVersion:
            continue
        files = [f for f in files if not f.get("yanked")]
        if parsed.is_prerelease or parsed.is_devrelease or not files:
            continue
        out[v] = min(f["upload_time"][:10] for f in files)
    return dict(sorted(out.items(), key=lambda kv: Version(kv[0])))


def fetch_releases(library: str, out_dir: Path = PYPI_DIR) -> dict[str, str]:
    """Download the release-date table for `library` and write it to out_dir/<library>.json."""
    req = urllib.request.Request(f"https://pypi.org/pypi/{library}/json", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        releases = release_dates(json.load(resp)["releases"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{library}.json").write_text(json.dumps(releases, indent=1), encoding="utf-8")
    return releases


def load_releases(library: str, pypi_dir: Path = PYPI_DIR) -> dict[str, str]:
    return json.loads((pypi_dir / f"{library}.json").read_text(encoding="utf-8"))


def version_at(releases: dict[str, str], day: date) -> str | None:
    """Highest version already released on `day` (so 1.5.3 shipped after 2.0.0 doesn't win), None if none was."""
    out = [v for v, d in releases.items() if date.fromisoformat(d) <= day]
    return max(out, key=Version) if out else None
