import importlib.util
import sys
from pathlib import Path

from snakepp import cache
from snakepp.generation import llm


def _app():
    spec = importlib.util.spec_from_file_location("demo_app", Path("demo/app.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_versions():
    assert _app().parse_versions("pandas==1.5.3, numpy 1.26") == {"pandas": "1.5.3", "numpy": "1.26"}


def test_answer_without_retrieval(tmp_path, monkeypatch):
    app = _app()
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    prompts = []
    monkeypatch.setattr(llm, "call_backend", lambda p, s, c: prompts.append(p) or "```python\nx = 1\n```")
    code, sources, raw = app.answer("Add a row on pandas 2.2", "", "No retrieval")
    assert code == "x = 1" and sources == "_no retrieval_" and "pandas 2.2" in prompts[0]


def test_source_list(chunks_200):
    text = _app().source_list(chunks_200[:2], cited=[2])
    assert text.startswith("[1] [") and "**cited**" in text.splitlines()[1]
