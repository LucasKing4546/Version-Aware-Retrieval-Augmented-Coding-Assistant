"""Version resolver. Owner: Kevin."""

from __future__ import annotations


def detect_versions(text: str) -> dict[str, str]:
    """Target versions found in a question, requirements file or import block: {"pandas": "1.5.3"}."""
    raise NotImplementedError("Kevin: version resolver (Week 9)")


def list_indexed_versions(library: str | None = None) -> dict[str, list[str]]:
    """Libraries and the versions indexed for each, from data/manifest.json."""
    raise NotImplementedError("Kevin")
