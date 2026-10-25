"""LLM wrapper — owner: Enea.

One generate() for three backends: vllm (any OpenAI-compatible server: vLLM on Kaggle/Colab, LM Studio),
openai and anthropic. Every call is cached in data/cache/llm keyed by prompt + system + config, so re-runs
are free. Endpoints and keys come from the environment (.env), never from configs.
"""

from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv

from snakepp import cache
from snakepp.config import GenerationConfig


def _call_openai_compatible(prompt: str, system: str, cfg: GenerationConfig, base_url: str | None) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY") or "not-needed")
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(prompt: str, system: str, cfg: GenerationConfig) -> str:
    from anthropic import Anthropic

    resp = Anthropic().messages.create(
        model=cfg.model,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def call_backend(prompt: str, system: str, cfg: GenerationConfig) -> str:
    load_dotenv()
    if cfg.backend == "vllm":
        base_url = os.environ.get("VLLM_BASE_URL")
        if not base_url:
            raise RuntimeError("set VLLM_BASE_URL in .env, e.g. http://localhost:1234/v1 for LM Studio")
        return _call_openai_compatible(prompt, system, cfg, base_url)
    if cfg.backend == "openai":
        return _call_openai_compatible(prompt, system, cfg, None)
    return _call_anthropic(prompt, system, cfg)


def complete(prompt: str, cfg: GenerationConfig, system: str = "") -> dict[str, Any]:
    """{'text', 'latency_s', 'cached'} for one prompt; served from data/cache/llm when seen before."""
    calls: list[float] = []

    def compute() -> dict[str, Any]:
        start = time.perf_counter()
        text = call_backend(prompt, system, cfg)
        calls.append(time.perf_counter() - start)
        return {"text": text, "latency_s": round(calls[-1], 3)}

    out = cache.cached_json("llm", cache.key(prompt, system, cfg.model_dump()), compute)
    return {**out, "cached": not calls}


def generate(prompt: str, cfg: GenerationConfig, system: str = "") -> str:
    return complete(prompt, cfg, system)["text"]
