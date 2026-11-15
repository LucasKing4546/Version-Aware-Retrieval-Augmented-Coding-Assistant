"""Version resolver: which versions a question targets, and whether a chunk fits them. Owner: Kevin."""

from __future__ import annotations

import re
from typing import Literal

from snakepp.schemas import Chunk

Compat = Literal["match", "mismatch", "unknown"]

# Names people write -> the library name chunks use.
ALIASES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scikit-learn": "scikit-learn",
    "scikit_learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "python": "python",
}
_NAME = "|".join(sorted((re.escape(a) for a in ALIASES), key=len, reverse=True))
_VERSION = r"\d+(?:\.\d+)*"
# "pandas==1.5.3", "pandas ~= 2.2", "pandas===2.2.3": an exact pin. Ranges (>=, <, !=) are not a target.
_PIN = re.compile(rf"(?<![\w-])({_NAME})\s*(?:===?|~=)\s*({_VERSION})(?![\w.])", re.I)
# "pandas 2.2.3", "pandas version 1.5", "pandas v2.0", "python3.11": prose needs a dotted version.
_PROSE = re.compile(rf"(?<![\w-])({_NAME})\s*(?:version\s+|v)?(\d+\.\d+(?:\.\d+)*)(?![\w.])", re.I)


def detect_versions(text: str) -> dict[str, str]:
    """Target versions named in a question, requirements file or import block: {"pandas": "1.5.3"}.

    Exact pins win over prose; within each kind the first mention wins. Unpinned libraries are left out.
    """
    found: dict[str, str] = {}
    for pattern in (_PIN, _PROSE):
        for match in pattern.finditer(text):
            found.setdefault(ALIASES[match.group(1).lower()], match.group(2))
    return found


def major(version: str | None) -> str | None:
    return version.split(".")[0] if version else None


def compatibility(chunk: Chunk, target_versions: dict[str, str]) -> Compat:
    """Same major version as the target -> match. No chunk version or no target for its library -> unknown."""
    target = major(target_versions.get(chunk.library))
    if target is None or chunk.version is None:
        return "unknown"
    return "match" if major(chunk.version) == target else "mismatch"


def list_indexed_versions(library: str | None = None) -> dict[str, list[str]]:
    """Libraries and the versions indexed for each, from data/manifest.json."""
    raise NotImplementedError("Kevin: needs Lucas's manifest (Week 10)")
