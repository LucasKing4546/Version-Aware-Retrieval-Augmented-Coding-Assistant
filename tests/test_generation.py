import pytest

from snakepp import cache
from snakepp.config import GenerationConfig
from snakepp.generation import llm
from snakepp.generation.prompts import build_prompt, format_chunk
from snakepp.schemas import Task

TASK = Task(task_id="t1", benchmark="version_set", prompt="Append a row to df.", target_versions={"pandas": "2.2.2"})


def test_no_context_prompt():
    p = build_prompt(TASK, [], GenerationConfig())
    assert p.startswith("Installed library versions: pandas 2.2.2.")
    assert "Append a row to df." in p and "Sources" not in p


def test_versions_line_can_be_disabled():
    assert "Installed" not in build_prompt(TASK, [], GenerationConfig(mention_versions=False))


def test_context_prompt_numbers_chunks(chunks_200):
    p = build_prompt(TASK, chunks_200[:3], GenerationConfig())
    assert "[1] " in p and "[3] " in p and "[4] " not in p
    assert "Sources: [n], [m]" in p


def test_format_chunk_truncates(chunks_200):
    so = next(c for c in chunks_200 if c.source == "stackoverflow" and c.version is None)
    text = format_chunk(2, so, max_chars=20)
    assert text.startswith("[2] ") and "unknown version, Stack Overflow" in text and text.endswith(" ...")


def test_complete_caches(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "call_backend", lambda p, s, c: calls.append(p) or "```python\nx = 1\n```")
    cfg = GenerationConfig()
    first = llm.complete("prompt", cfg)
    second = llm.complete("prompt", cfg)
    assert first["cached"] is False and second["cached"] is True
    assert second["text"] == first["text"] and len(calls) == 1
    assert llm.complete("prompt", GenerationConfig(temperature=0.5))["cached"] is False


def test_vllm_needs_base_url(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="VLLM_BASE_URL"):
        llm.call_backend("p", "", GenerationConfig())
