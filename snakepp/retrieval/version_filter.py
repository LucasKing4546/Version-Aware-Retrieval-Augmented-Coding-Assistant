"""Version filter — owner: Kevin.

Target versions come from (highest priority first) an explicit argument, the query text ("pandas 1.5",
"numpy==2.0") or, optionally, the libraries installed in the current environment (importlib.metadata).
"""

from __future__ import annotations

import re
from importlib import metadata

from packaging.version import InvalidVersion, Version

from snakepp.schemas import Chunk, Retrieved


def compatible(chunk: Chunk, target_versions: dict[str, str], max_minor_gap: int | None = None) -> bool:
    """Compatible when the library is not targeted, the chunk has no version, or the major versions match
    (and, with max_minor_gap, the minor versions are at most that far apart)."""
    target = target_versions.get(chunk.library)
    if target is None or chunk.version is None:
        return True
    try:
        have, want = Version(chunk.version), Version(target)
    except InvalidVersion:
        return True
    if have.major != want.major:
        return False
    return max_minor_gap is None or abs(have.minor - want.minor) <= max_minor_gap


def apply(
    results: list[Retrieved],
    chunks: dict[str, Chunk],
    target_versions: dict[str, str],
    mode: str = "soft",
    penalty: float = 0.3,
    max_minor_gap: int | None = None,
) -> list[Retrieved]:
    """hard: drop incompatible chunks; soft: multiply their score by `penalty`; off: unchanged. Re-ranks."""
    if mode == "off":
        return results
    kept: list[tuple[str, float, str]] = []
    for r in results:
        ok = compatible(chunks[r.chunk_id], target_versions, max_minor_gap)
        if ok:
            kept.append((r.chunk_id, r.score, r.stage))
        elif mode == "soft":
            kept.append((r.chunk_id, r.score * penalty, r.stage))
    kept.sort(key=lambda x: -x[1])
    return [Retrieved(chunk_id=c, rank=i + 1, score=s, stage=st) for i, (c, s, st) in enumerate(kept)]


LIBRARIES = ("pandas", "numpy", "matplotlib", "scikit-learn", "scipy")
_ALIASES = {"sklearn": "scikit-learn", "np": "numpy", "pd": "pandas"}
_LIB = r"\b(pandas|numpy|matplotlib|scikit-learn|sklearn|scipy)\b"
# 'pandas 1.5', 'pandas==2.2.2', 'numpy version 2', 'sklearn v1.4' -- but not 'pandas 4 columns'
_MENTION = re.compile(
    _LIB + r"\s*(?:(?:==|>=|<=|~=)\s*|version\s+|v)(\d+(?:\.\d+){0,2})\b|"
    + _LIB + r"\s+(\d+\.\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def versions_in_text(text: str) -> dict[str, str]:
    """Library versions named in free text, e.g. 'I am on pandas 1.5' -> {'pandas': '1.5'}."""
    found: dict[str, str] = {}
    for lib_a, ver_a, lib_b, ver_b in _MENTION.findall(text):
        lib, version = (lib_a, ver_a) if lib_a else (lib_b, ver_b)
        if int(version.split(".")[0]) < 10:  # our libraries never reached major 10
            found.setdefault(_ALIASES.get(lib.lower(), lib.lower()), version)
    return found


def installed_versions(libraries: tuple[str, ...] = LIBRARIES) -> dict[str, str]:
    found = {}
    for lib in libraries:
        try:
            found[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            pass
    return found


def detect_versions(query: str, explicit: dict[str, str] | None = None, use_installed: bool = False) -> dict[str, str]:
    """Merge installed (lowest priority) < mentioned in the query < explicit."""
    versions = installed_versions() if use_installed else {}
    versions.update(versions_in_text(query))
    versions.update(explicit or {})
    return versions
