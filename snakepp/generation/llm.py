"""llm — owner: Enea, due: M0 Mon 26 Oct.

TODO:
One generate(prompt, cfg: GenerationConfig) -> str for the vllm (OpenAI-compatible endpoint), anthropic and
openai backends. Cache every call with snakepp.cache keyed by prompt + config. Log latency.
"""

from __future__ import annotations
