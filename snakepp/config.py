"""Experiment configs (configs/*.yaml) and pinned versions (configs/pins.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from snakepp.schemas import Source

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class RetrievalConfig(BaseModel):
    mode: Literal["none", "bm25", "dense", "hybrid"] = "hybrid"
    sources: list[Source] = Field(default_factory=lambda: ["docs", "stackoverflow"])
    k: int = 5
    rerank: bool = True
    corpus: Literal["clean", "leaky"] = "clean"  # Stack Overflow variant
    # Knobs below are not contract keys; defaults come from ARCHITECTURE.md §9.
    dense_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    candidates: int = 50
    rrf_k: int = 60


class VersionFilterConfig(BaseModel):
    mode: Literal["off", "soft", "hard"] = "off"
    penalty: float = 0.3  # score multiplier for an incompatible chunk in soft mode


class GenerationConfig(BaseModel):
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    backend: Literal["vllm", "openai", "anthropic"] = "vllm"
    temperature: float = 0.0
    max_tokens: int = 1024


class Config(BaseModel):
    name: str
    description: str = ""
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    version_filter: VersionFilterConfig = Field(default_factory=VersionFilterConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class Pins(BaseModel):
    """Target library versions per sandbox environment, plus its Python version."""

    environments: dict[str, dict[str, str]]  # {"ds1000": {"python": "3.10", "pandas": "1.3.5", ...}}


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.suffix:
        path = CONFIGS_DIR / f"{path}.yaml"
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))


def load_pins(path: str | Path = CONFIGS_DIR / "pins.yaml") -> Pins:
    with open(path, encoding="utf-8") as f:
        return Pins.model_validate(yaml.safe_load(f))
