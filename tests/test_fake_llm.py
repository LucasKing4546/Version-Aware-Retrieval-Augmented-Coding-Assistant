"""The LLM cache key and FakeLLM (recorded and scripted replies)."""

from __future__ import annotations

import inspect

import pytest

from snakepp.config import GenerationConfig
from snakepp.generation import llm
from snakepp.generation.llm import LLMRecord, cache_key
from tests.fakes.fake_llm import FakeLLM, UnrecordedCall

CFG = GenerationConfig()
MSGS = [
    {"role": "system", "content": "You write Python."},
    {"role": "user", "content": "Append a row to a DataFrame in pandas 2.2.2."},
]


class TestCacheKey:
    def test_is_deterministic_and_hex(self):
        key = cache_key(MSGS, CFG)
        assert key == cache_key([dict(m) for m in MSGS], GenerationConfig())
        assert len(key) == 64 and int(key, 16) >= 0

    def test_ignores_dict_key_order(self):
        reordered = [{"content": m["content"], "role": m["role"]} for m in MSGS]
        assert cache_key(reordered, CFG) == cache_key(MSGS, CFG)

    @pytest.mark.parametrize(
        "cfg",
        [
            GenerationConfig(model="Qwen/Qwen2.5-Coder-1.5B-Instruct"),
            GenerationConfig(temperature=0.2),
            GenerationConfig(max_tokens=512),
            GenerationConfig(backend="openai"),
        ],
        ids=["model", "temperature", "max_tokens", "backend"],
    )
    def test_changes_with_config(self, cfg):
        assert cache_key(MSGS, cfg) != cache_key(MSGS, CFG)

    def test_changes_with_messages(self):
        other = [*MSGS[:-1], {"role": "user", "content": "Append a row in pandas 1.5.3."}]
        assert cache_key(other, CFG) != cache_key(MSGS, CFG)


def test_fake_has_the_real_signature():
    fake_params = list(inspect.signature(FakeLLM().complete).parameters)
    assert fake_params == list(inspect.signature(llm.complete).parameters) == ["messages", "cfg"]


def test_replays_a_recording():
    fake = FakeLLM()
    fake.record(MSGS, CFG, "df = pd.concat([df, row], ignore_index=True)")
    assert fake.complete(MSGS, CFG) == "df = pd.concat([df, row], ignore_index=True)"
    assert fake.calls == [MSGS]


def test_recording_is_tied_to_the_config():
    fake = FakeLLM()
    fake.record(MSGS, CFG, "7B reply")
    with pytest.raises(UnrecordedCall, match="Qwen2.5-Coder-1.5B"):
        fake.complete(MSGS, GenerationConfig(model="Qwen/Qwen2.5-Coder-1.5B-Instruct"))


def test_script_is_used_in_order_and_can_be_a_function():
    fake = FakeLLM(script=["first", lambda messages: f"saw {len(messages)} messages"])
    assert fake.complete(MSGS, CFG) == "first"
    assert fake.complete(MSGS, CFG) == "saw 2 messages"
    assert fake.remaining == 0


def test_recording_wins_over_script():
    fake = FakeLLM(script=["scripted"])
    fake.record(MSGS, CFG, "recorded")
    assert fake.complete(MSGS, CFG) == "recorded"
    assert fake.remaining == 1


def test_unknown_call_raises_with_context():
    fake = FakeLLM(script=["only one"])
    fake.complete(MSGS, CFG)
    with pytest.raises(UnrecordedCall, match=r"(?s)call 2 .*Something new in pandas 2\.2\.2"):
        fake.complete(MSGS[:1] + [{"role": "user", "content": "Something new in pandas 2.2.2"}], CFG)


def test_calls_are_copied():
    fake = FakeLLM(script=["x"])
    msgs = [dict(m) for m in MSGS]
    fake.complete(msgs, CFG)
    msgs[0]["content"] = "mutated later"
    assert fake.calls[0][0]["content"] == "You write Python."


def test_jsonl_round_trip(tmp_path):
    fake = FakeLLM()
    fake.record(MSGS, CFG, "a")
    fake.record(MSGS[:1], CFG, "b")
    path = tmp_path / "llm" / "replies.jsonl"
    fake.save(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [LLMRecord.model_validate_json(x).key for x in lines] == sorted(fake.records)
    loaded = FakeLLM.from_jsonl(path)
    assert loaded.complete(MSGS, CFG) == "a" and loaded.complete(MSGS[:1], CFG) == "b"


def test_install_replaces_the_module_function(monkeypatch):
    fake = FakeLLM(script=["from the fake"]).install(monkeypatch)
    assert llm.complete(MSGS, CFG) == "from the fake"
    assert fake.calls == [MSGS]


def test_real_stub_is_restored_after_install():
    with pytest.raises(NotImplementedError):
        llm.complete(MSGS, CFG)
