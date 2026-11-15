"""llm.complete: cache behaviour and the three backends, with no network (clients are faked)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from snakepp.config import GenerationConfig
from snakepp.generation import llm
from snakepp.generation.llm import LLMRefusal, Reply

MSGS = [
    {"role": "system", "content": "You write Python."},
    {"role": "user", "content": "Append a row to a DataFrame in pandas 2.2.2."},
]
VLLM = GenerationConfig()
OPENAI = GenerationConfig(backend="openai", model="some-openai-model")
CLAUDE = GenerationConfig(backend="anthropic", model="claude-opus-5-5")


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    """Every test gets an empty cache; nothing is written under data/."""
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm")
    return tmp_path / "llm"


class CountingBackend:
    def __init__(self, text: str = "df = pd.concat([df, row])") -> None:
        self.text = text
        self.calls = 0

    def __call__(self, messages, cfg) -> Reply:
        self.calls += 1
        return Reply(text=f"{self.text} #{self.calls}", prompt_tokens=10, completion_tokens=5, finish_reason="stop")


@pytest.fixture
def backend(monkeypatch) -> CountingBackend:
    fake = CountingBackend()
    for name in ("vllm", "openai", "anthropic"):
        monkeypatch.setitem(llm._BACKENDS, name, fake)
    return fake


# ---- cache -------------------------------------------------------------------------------------


def test_second_call_is_served_from_the_cache(backend, cache_dir):
    first = llm.complete(MSGS, VLLM)
    assert llm.complete(MSGS, VLLM) == first
    assert backend.calls == 1
    key = llm.cache_key(MSGS, VLLM)
    assert llm.cache_path(key) == cache_dir / key[:2] / f"{key}.json"
    assert llm.cache_path(key).is_file()


def test_record_keeps_usage_and_latency(backend):
    rec = llm.complete_record(MSGS, VLLM)
    assert (rec.prompt_tokens, rec.completion_tokens, rec.finish_reason) == (10, 5, "stop")
    assert rec.latency_s is not None and rec.latency_s >= 0
    assert rec.model == VLLM.model and rec.messages == MSGS


def test_a_different_config_is_a_different_entry(backend):
    llm.complete(MSGS, VLLM)
    llm.complete(MSGS, GenerationConfig(model="Qwen/Qwen2.5-Coder-1.5B-Instruct"))
    assert backend.calls == 2


def test_corrupt_entry_is_a_miss_and_gets_replaced(backend):
    path = llm.cache_path(llm.cache_key(MSGS, VLLM))
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    reply = llm.complete(MSGS, VLLM)
    assert backend.calls == 1
    assert llm.LLMRecord.model_validate_json(path.read_text(encoding="utf-8")).reply == reply


def test_no_temp_files_are_left_behind(backend, cache_dir):
    llm.complete(MSGS, VLLM)
    assert not [p for p in cache_dir.rglob("*") if p.name.startswith(".tmp-")]


def test_a_cache_entry_is_a_valid_recording(backend, cache_dir):
    from tests.fakes.fake_llm import FakeLLM

    reply = llm.complete(MSGS, VLLM)
    (entry,) = cache_dir.rglob("*.json")
    fake = FakeLLM([llm.LLMRecord.model_validate_json(entry.read_text(encoding="utf-8"))])
    assert fake.complete(MSGS, VLLM) == reply


def test_refusal_is_not_cached(monkeypatch, cache_dir):
    def refuse(messages, cfg):
        raise LLMRefusal("declined")

    monkeypatch.setitem(llm._BACKENDS, "anthropic", refuse)
    with pytest.raises(LLMRefusal):
        llm.complete(MSGS, CLAUDE)
    assert not list(cache_dir.rglob("*.json"))


def test_empty_messages_rejected(backend):
    with pytest.raises(ValueError, match="empty"):
        llm.complete([], VLLM)
    assert backend.calls == 0


# ---- OpenAI-compatible backends (vllm, openai) ------------------------------------------------


class FakeOpenAI:
    def __init__(self, content: str | None = "x = 1") -> None:
        self.kwargs: dict = {}
        self.content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
        )


@pytest.fixture
def openai_client(monkeypatch) -> FakeOpenAI:
    client = FakeOpenAI()
    monkeypatch.setattr(llm, "_openai_client", lambda backend: client)
    return client


def test_vllm_request(openai_client):
    rec = llm.complete_record(MSGS, VLLM)
    assert openai_client.kwargs == {
        "model": VLLM.model,
        "messages": MSGS,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    assert (rec.reply, rec.prompt_tokens, rec.completion_tokens, rec.finish_reason) == ("x = 1", 12, 3, "stop")


def test_openai_uses_max_completion_tokens(openai_client):
    llm.complete(MSGS, OPENAI)
    assert openai_client.kwargs["max_completion_tokens"] == 1024 and "max_tokens" not in openai_client.kwargs


def test_empty_content_becomes_empty_string(monkeypatch):
    monkeypatch.setattr(llm, "_openai_client", lambda backend: FakeOpenAI(content=None))
    assert llm.complete(MSGS, VLLM) == ""


def test_vllm_needs_a_base_url(monkeypatch):
    llm._openai_client.cache_clear()
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="VLLM_BASE_URL"):
        llm.complete(MSGS, VLLM)
    llm._openai_client.cache_clear()


def test_vllm_client_points_at_the_base_url(monkeypatch):
    llm._openai_client.cache_clear()
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    client = llm._openai_client("vllm")
    assert str(client.base_url).rstrip("/") == "http://localhost:8000/v1"
    llm._openai_client.cache_clear()


# ---- Anthropic backend -----------------------------------------------------------------------


class FakeAnthropic:
    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.kwargs: dict = {}
        self.stop_reason = stop_reason
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="df = pd.concat("),
                SimpleNamespace(type="text", text="[df, row])"),
            ],
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(category="cyber") if self.stop_reason == "refusal" else None,
            usage=SimpleNamespace(input_tokens=20, output_tokens=8),
        )


def test_anthropic_request_and_text(monkeypatch):
    client = FakeAnthropic()
    monkeypatch.setattr(llm, "_anthropic_client", lambda: client)
    rec = llm.complete_record(MSGS, CLAUDE)

    assert client.kwargs == {
        "model": "claude-opus-5-5",
        "max_tokens": 1024,
        "system": "You write Python.",
        "messages": MSGS[1:],
    }
    assert "temperature" not in client.kwargs  # rejected by API version 2026-10-21
    assert rec.reply == "df = pd.concat([df, row])"  # text blocks joined, thinking skipped
    assert (rec.prompt_tokens, rec.completion_tokens, rec.finish_reason) == (20, 8, "end_turn")


def test_anthropic_without_system(monkeypatch):
    client = FakeAnthropic()
    monkeypatch.setattr(llm, "_anthropic_client", lambda: client)
    llm.complete(MSGS[1:], CLAUDE)
    assert "system" not in client.kwargs


def test_anthropic_refusal_raises(monkeypatch, cache_dir):
    monkeypatch.setattr(llm, "_anthropic_client", lambda: FakeAnthropic(stop_reason="refusal"))
    with pytest.raises(LLMRefusal, match="cyber"):
        llm.complete(MSGS, CLAUDE)
    assert not list(cache_dir.rglob("*.json"))


@pytest.mark.parametrize(
    "messages,error",
    [
        ([{"role": "system", "content": "s"}], "no user message"),
        (
            [{"role": "user", "content": "u"}, {"role": "system", "content": "late"}],
            "only supported at the start",
        ),
    ],
)
def test_system_message_placement(messages, error):
    with pytest.raises(ValueError, match=error):
        llm._split_system(messages)


def test_several_leading_system_messages_are_joined():
    system, turns = llm._split_system([{"role": "system", "content": "a"}, {"role": "system", "content": "b"}, MSGS[1]])
    assert system == "a\n\nb" and turns == [MSGS[1]]
