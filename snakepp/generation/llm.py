"""Backend-agnostic LLM client; every call cached in data/cache/. Owner: Enea."""

from __future__ import annotations

from snakepp.config import GenerationConfig


def complete(messages: list[dict[str, str]], cfg: GenerationConfig) -> str:
    """OpenAI-style messages in, completion text out. Cached by a hash of messages + cfg."""
    raise NotImplementedError("Enea: LLM client with cache (Week 8)")
