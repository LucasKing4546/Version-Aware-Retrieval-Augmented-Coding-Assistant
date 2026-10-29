"""Version filter — owner: Kevin, due: M3 Thu 5 Nov.

TODO: version detection from query text, explicit argument or importlib.metadata.
"""

from __future__ import annotations

from packaging.version import InvalidVersion, Version

from snakepp.schemas import Chunk, Retrieved


def compatible(chunk: Chunk, target_versions: dict[str, str]) -> bool:
    """A chunk is compatible when its library is not targeted, it has no version, or the major versions match."""
    target = target_versions.get(chunk.library)
    if target is None or chunk.version is None:
        return True
    try:
        return Version(chunk.version).major == Version(target).major
    except InvalidVersion:
        return True


def apply(
    results: list[Retrieved],
    chunks: dict[str, Chunk],
    target_versions: dict[str, str],
    mode: str = "soft",
    penalty: float = 0.3,
) -> list[Retrieved]:
    """hard: drop incompatible chunks; soft: multiply their score by `penalty`; off: unchanged. Re-ranks."""
    if mode == "off":
        return results
    kept: list[tuple[str, float, str]] = []
    for r in results:
        ok = compatible(chunks[r.chunk_id], target_versions)
        if ok:
            kept.append((r.chunk_id, r.score, r.stage))
        elif mode == "soft":
            kept.append((r.chunk_id, r.score * penalty, r.stage))
    kept.sort(key=lambda x: -x[1])
    return [Retrieved(chunk_id=c, rank=i + 1, score=s, stage=st) for i, (c, s, st) in enumerate(kept)]
