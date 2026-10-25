"""Experiment configuration loaded from configs/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    enabled: bool = True
    sources: list[Literal["docs", "stackoverflow"]] = Field(default_factory=lambda: ["docs", "stackoverflow"])
    bm25: bool = True
    dense: bool = True
    dense_model: str = "BAAI/bge-small-en-v1.5"
    rrf_k: int = 60
    candidates: int = 50
    rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    top_k: int = 3
    so_index: Literal["clean", "leaky"] = "clean"


class VersionFilterConfig(BaseModel):
    mode: Literal["off", "soft", "hard"] = "off"
    penalty: float = 0.3  # score multiplier for incompatible chunks in soft mode


class GenerationConfig(BaseModel):
    backend: Literal["vllm", "anthropic", "openai"] = "vllm"
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 1024


class Config(BaseModel):
    name: str
    description: str = ""
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    version_filter: VersionFilterConfig = Field(default_factory=VersionFilterConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
