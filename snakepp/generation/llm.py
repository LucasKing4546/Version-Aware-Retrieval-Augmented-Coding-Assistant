"""Backend-agnostic LLM client; every call cached in data/cache/llm/. Owner: Enea.

Backends (`GenerationConfig.backend`):
- `vllm`: an open model behind vLLM's OpenAI-compatible server. Set `VLLM_BASE_URL` (for example
  `http://localhost:8000/v1`); `VLLM_API_KEY` is optional.
- `openai`: the OpenAI API (`OPENAI_API_KEY`).
- `anthropic`: the Claude Messages API (`ANTHROPIC_API_KEY` or an `ant auth login` profile). The
  installed SDK sends API version 2026-10-21, which rejects `temperature`, so it is not sent:
  Claude samples at its defaults and `cfg.temperature` has no effect on this backend.

Every successful call is stored as an `LLMRecord` under `data/cache/llm/<key[:2]>/<key>.json`, so
reruns and resumed sweeps cost nothing. A refusal raises `LLMRefusal` and is not cached. No
fallback model is used: in an experiment the reply must come from the configured model.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from snakepp import paths
from snakepp.config import GenerationConfig

CACHE_DIR = paths.CACHE / "llm"
TIMEOUT_S = 300.0  # per request; the SDKs retry connection errors, 429 and 5xx twice
Messages = list[dict[str, str]]


class LLMRecord(BaseModel):
    """One cached or recorded call. The cache (data/cache/) and the test recordings
    (tests/fixtures/) share this format, so a cache entry can be committed as a recording."""

    key: str
    model: str
    messages: Messages
    reply: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None  # "stop", "length", "end_turn", "max_tokens", ...
    latency_s: float | None = None  # of the original call, not of a cache hit


class LLMRefusal(RuntimeError):
    """The model declined the request (Claude `stop_reason == "refusal"`)."""


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


def cache_key(messages: Messages, cfg: GenerationConfig) -> str:
    """Hash of the messages and the whole generation config (model, backend, temperature...)."""
    payload = {"messages": messages, "cfg": cfg.model_dump(mode="json")}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_record(messages: Messages, cfg: GenerationConfig, reply: str) -> LLMRecord:
    return LLMRecord(key=cache_key(messages, cfg), model=cfg.model, messages=messages, reply=reply)


def complete(messages: list[dict[str, str]], cfg: GenerationConfig) -> str:
    """OpenAI-style messages in, completion text out. Cached by a hash of messages + cfg."""
    return complete_record(messages, cfg).reply


def complete_record(messages: Messages, cfg: GenerationConfig) -> LLMRecord:
    """Like `complete`, but returns the whole record (token counts, finish reason, latency)."""
    if not messages:
        raise ValueError("messages is empty")
    key = cache_key(messages, cfg)
    path = cache_path(key)
    if (hit := _read(path)) is not None:
        return hit

    start = time.perf_counter()
    reply = _BACKENDS[cfg.backend](messages, cfg)
    record = LLMRecord(
        key=key,
        model=cfg.model,
        messages=messages,
        reply=reply.text,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        finish_reason=reply.finish_reason,
        latency_s=round(time.perf_counter() - start, 3),
    )
    _write(path, record)
    return record


def cache_path(key: str) -> Path:
    return CACHE_DIR / key[:2] / f"{key}.json"


# ---- cache files -------------------------------------------------------------------------------


def _read(path: Path) -> LLMRecord | None:
    try:
        return LLMRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (ValidationError, ValueError, OSError):
        return None  # a corrupt or half-written entry is a miss; the new reply overwrites it


def _write(path: Path, record: LLMRecord) -> None:
    """Atomic: write a temp file next to the target, then rename, so parallel workers never
    see a partial entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(record.model_dump_json())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---- backends ----------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _openai_client(backend: str) -> Any:
    try:
        import openai
    except ImportError as exc:
        raise ImportError('backend "vllm"/"openai" needs the gen extra: pip install -e ".[gen]"') from exc
    from dotenv import load_dotenv

    load_dotenv()
    if backend == "vllm":
        base_url = os.environ.get("VLLM_BASE_URL")
        if not base_url:
            raise RuntimeError("backend vllm: set VLLM_BASE_URL (e.g. http://localhost:8000/v1) in .env")
        return openai.OpenAI(base_url=base_url, api_key=os.environ.get("VLLM_API_KEY") or "EMPTY", timeout=TIMEOUT_S)
    return openai.OpenAI(timeout=TIMEOUT_S)


@lru_cache(maxsize=1)
def _anthropic_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError('backend "anthropic" needs the gen extra: pip install -e ".[gen]"') from exc
    from dotenv import load_dotenv

    load_dotenv()
    return anthropic.Anthropic(timeout=TIMEOUT_S)


def _call_openai_compatible(messages: Messages, cfg: GenerationConfig) -> Reply:
    # vLLM takes max_tokens; the OpenAI API wants max_completion_tokens for current models.
    limit = {"max_tokens": cfg.max_tokens} if cfg.backend == "vllm" else {"max_completion_tokens": cfg.max_tokens}
    response = _openai_client(cfg.backend).chat.completions.create(
        model=cfg.model, messages=messages, temperature=cfg.temperature, **limit
    )
    choice = response.choices[0]
    usage = response.usage
    return Reply(
        text=choice.message.content or "",
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        finish_reason=choice.finish_reason,
    )


def _call_anthropic(messages: Messages, cfg: GenerationConfig) -> Reply:
    system, turns = _split_system(messages)
    kwargs: dict[str, Any] = {"model": cfg.model, "max_tokens": cfg.max_tokens, "messages": turns}
    if system:
        kwargs["system"] = system
    response = _anthropic_client().messages.create(**kwargs)
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise LLMRefusal(f"{cfg.model} refused (category {getattr(details, 'category', None)})")
    # Thinking blocks (adaptive thinking is on by default on current models) are skipped.
    text = "".join(block.text for block in response.content if block.type == "text")
    return Reply(
        text=text,
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        finish_reason=response.stop_reason,
    )


def _split_system(messages: Messages) -> tuple[str, Messages]:
    """Leading system messages become the Messages API `system` field."""
    i = 0
    while i < len(messages) and messages[i]["role"] == "system":
        i += 1
    turns = messages[i:]
    if any(m["role"] == "system" for m in turns):
        raise ValueError("system messages are only supported at the start of the conversation")
    if not turns:
        raise ValueError("no user message after the system messages")
    return "\n\n".join(m["content"] for m in messages[:i]), turns


_BACKENDS: dict[str, Callable[[Messages, GenerationConfig], Reply]] = {
    "vllm": _call_openai_compatible,
    "openai": _call_openai_compatible,
    "anthropic": _call_anthropic,
}
