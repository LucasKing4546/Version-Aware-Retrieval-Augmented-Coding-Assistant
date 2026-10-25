"""Prompt templates — owner: Enea.

Two templates: no context (baseline) and context, where retrieved chunks are listed as [1]..[k] with their
library and version and the model is asked for one python code block plus a 'Sources: [n], [m]' line.
"""

from __future__ import annotations

from snakepp.config import GenerationConfig
from snakepp.schemas import Chunk, Task

SYSTEM = "You are an expert Python programmer. Answer with correct, runnable code."

ANSWER_FORMAT = "Reply with exactly one ```python code block and nothing else."
ANSWER_FORMAT_CITED = (
    "Reply with exactly one ```python code block, then one line 'Sources: [n], [m]' listing the context "
    "entries you used (write 'Sources: none' if none helped)."
)


def _versions_line(task: Task, cfg: GenerationConfig) -> str:
    if not (cfg.mention_versions and task.target_versions):
        return ""
    pins = ", ".join(f"{lib} {v}" for lib, v in sorted(task.target_versions.items()))
    return f"Installed library versions: {pins}. The code must work with exactly these versions.\n\n"


def format_chunk(n: int, chunk: Chunk, max_chars: int) -> str:
    version = chunk.version or "unknown version"
    origin = "docs" if chunk.source == "docs" else "Stack Overflow"
    text = chunk.text if len(chunk.text) <= max_chars else chunk.text[:max_chars].rstrip() + " ..."
    return f"[{n}] {chunk.title} ({chunk.library} {version}, {origin})\n{text}"


def build_prompt(task: Task, chunks: list[Chunk], cfg: GenerationConfig) -> str:
    """No-context prompt when `chunks` is empty, otherwise the numbered-context prompt."""
    head = _versions_line(task, cfg)
    if not chunks:
        return f"{head}{task.prompt}\n\n{ANSWER_FORMAT}"
    context = "\n\n".join(format_chunk(i, c, cfg.context_chars) for i, c in enumerate(chunks, start=1))
    return (
        f"{head}Context (may be incomplete or for other versions):\n\n{context}\n\n"
        f"Task:\n{task.prompt}\n\n{ANSWER_FORMAT_CITED}"
    )
