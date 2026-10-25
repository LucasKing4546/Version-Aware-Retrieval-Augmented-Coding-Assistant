"""Library version inference for Stack Overflow posts — owner: Lucas.

The version of a post is the release that was current on its date (PyPI upload times, cached in
data/cache/pypi), unless the text names a version explicitly ("pandas 0.25", "numpy==1.21").
"""

from __future__ import annotations

import json
import urllib.request
from bisect import bisect_right
from datetime import date, datetime

from packaging.version import InvalidVersion, Version

from snakepp import cache
from snakepp.retrieval.version_filter import versions_in_text


def _fetch_pypi(package: str) -> dict[str, str]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=60) as resp:
        releases = json.load(resp)["releases"]
    out = {}
    for version, files in releases.items():
        if files and not all(f.get("yanked") for f in files):
            out[version] = min(f["upload_time"] for f in files)[:10]
    return out


def release_dates(package: str) -> dict[str, str]:
    """{version: 'YYYY-MM-DD'} of non-yanked releases, cached per package."""
    return cache.cached_json("pypi", cache.key(package), lambda: _fetch_pypi(package))


class VersionTimeline:
    """Final (non-pre) releases of one package in date order."""

    def __init__(self, dates: dict[str, str]) -> None:
        rows = []
        for v, d in dates.items():
            try:
                parsed = Version(v)
            except InvalidVersion:
                continue
            if not (parsed.is_prerelease or parsed.is_devrelease):
                rows.append((date.fromisoformat(d), parsed, v))
        rows.sort()
        self._dates = [r[0] for r in rows]
        self._versions = [r[2] for r in rows]

    def at(self, when: date) -> str | None:
        """Latest release published on or before `when` (None before the first release)."""
        i = bisect_right(self._dates, when)
        return self._versions[i - 1] if i else None


_TIMELINES: dict[str, VersionTimeline] = {}


def timeline(package: str) -> VersionTimeline:
    if package not in _TIMELINES:
        _TIMELINES[package] = VersionTimeline(release_dates(package))
    return _TIMELINES[package]


def infer_version(library: str, text: str, created: date | datetime) -> tuple[str | None, bool]:
    """(version, inferred): an explicit mention in the text wins (inferred=False), else the release at that date."""
    if mentioned := versions_in_text(text).get(library):
        return mentioned, False
    when = created.date() if isinstance(created, datetime) else created
    return timeline(library).at(when), True
