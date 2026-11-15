"""Version filter: the last retrieval step, applied to the final ranked list. Owner: Kevin.

off  -> nothing changes.
soft -> an incompatible chunk's score is multiplied by `penalty`; a chunk whose version was only inferred (Stack
        Overflow, from its post date) gets the softer (1 + penalty) / 2, because the guess may be wrong.
hard -> incompatible chunks are dropped. If nothing compatible is left, fewer than k hits come back (E5 measures this).
Chunks of unknown compatibility are never penalised.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from snakepp.config import VersionFilterConfig
from snakepp.retrieval.bm25 import Hits
from snakepp.retrieval.versions import Compat, compatibility
from snakepp.schemas import Chunk

Action = Literal["kept", "penalised", "dropped"]


@dataclass(frozen=True)
class Decision:
    """What the filter did to one chunk, for the trace command and failure analysis."""

    compat: Compat
    action: Action
    factor: float = 1.0


def penalty_for(chunk: Chunk, penalty: float) -> float:
    return (1.0 + penalty) / 2 if chunk.version_inferred else penalty


def _scale(score: float, factor: float) -> float:
    return score * factor if score >= 0 else score / factor  # a penalty must never raise a negative score


def apply(
    hits: Hits,
    chunks: Mapping[str, Chunk],
    target_versions: dict[str, str],
    cfg: VersionFilterConfig,
) -> tuple[Hits, dict[str, Decision]]:
    """Filtered hits, best first, plus each chunk's decision. Ties keep the incoming order."""
    out: Hits = []
    decisions: dict[str, Decision] = {}
    for chunk_id, score in hits:
        chunk = chunks[chunk_id]
        compat = compatibility(chunk, target_versions)
        if cfg.mode == "off" or compat != "mismatch":
            decisions[chunk_id] = Decision(compat, "kept")
            out.append((chunk_id, score))
        elif cfg.mode == "hard":
            decisions[chunk_id] = Decision(compat, "dropped", 0.0)
        else:
            factor = penalty_for(chunk, cfg.penalty)
            decisions[chunk_id] = Decision(compat, "penalised", factor)
            out.append((chunk_id, _scale(score, factor)))
    out.sort(key=lambda hit: -hit[1])
    return out, decisions
