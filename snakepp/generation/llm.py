"""Backend-agnostic LLM client; every call cached in data/cache/. Owner: Enea."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel

from snakepp.config import GenerationConfig


class LLMRecord(BaseModel):
    """One cached or recorded call. The cache (data/cache/) and the test recordings
    (tests/fixtures/) share this format, so a cache entry can be committed as a recording."""

    key: str
    model: str
    messages: list[dict[str, str]]
    reply: str


def cache_key(messages: list[dict[str, str]], cfg: GenerationConfig) -> str:
    """Hash of the messages and the whole generation config (model, backend, temperature...)."""
    payload = {"messages": messages, "cfg": cfg.model_dump(mode="json")}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_record(messages: list[dict[str, str]], cfg: GenerationConfig, reply: str) -> LLMRecord:
    return LLMRecord(key=cache_key(messages, cfg), model=cfg.model, messages=messages, reply=reply)


def complete(messages: list[dict[str, str]], cfg: GenerationConfig) -> str:
    """OpenAI-style messages in, completion text out. Cached by a hash of messages + cfg."""
    raise NotImplementedError("Enea: LLM client with cache (Week 8)")
