"""FakeLLM: recorded and scripted replies in place of `snakepp.generation.llm.complete`. Owner: Enea.

For each call the fake answers with, in order:
1. the recording whose key is `llm.cache_key(messages, cfg)`: an exact replay of an earlier call;
2. otherwise the next scripted reply: a string, or a function of the messages.
Anything else raises `UnrecordedCall`, so a test never gets an answer nobody wrote.

Recordings use the LLM cache's format (`llm.LLMRecord`), so cache entries from a real run can be
committed under tests/fixtures/ and replayed without a GPU.

Swapping it in is one line in a test: `fake = FakeLLM(script=[...]).install(monkeypatch)`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from snakepp.config import GenerationConfig
from snakepp.generation import llm
from snakepp.generation.llm import LLMRecord, cache_key, make_record

Messages = list[dict[str, str]]
Reply = str | Callable[[Messages], str]


class UnrecordedCall(LookupError):
    """The fake has neither a recording nor a scripted reply for this call."""


class FakeLLM:
    def __init__(self, records: Iterable[LLMRecord] = (), script: Iterable[Reply] = ()) -> None:
        self.records: dict[str, LLMRecord] = {r.key: r for r in records}
        self.script: deque[Reply] = deque(script)
        self.calls: list[Messages] = []

    @classmethod
    def from_jsonl(cls, path: str | Path, script: Iterable[Reply] = ()) -> FakeLLM:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls([LLMRecord.model_validate_json(line) for line in lines if line.strip()], script)

    def complete(self, messages: Messages, cfg: GenerationConfig) -> str:
        """Same signature as `llm.complete`."""
        self.calls.append([dict(m) for m in messages])
        key = cache_key(messages, cfg)
        if key in self.records:
            return self.records[key].reply
        if self.script:
            reply = self.script.popleft()
            return reply(messages) if callable(reply) else reply
        last = messages[-1] if messages else {}
        preview = str(last.get("content", ""))[:200]
        raise UnrecordedCall(
            f"no recording or scripted reply for call {len(self.calls)} (model {cfg.model}, key {key[:12]}).\n"
            f"Last message ({last.get('role', '?')}): {preview!r}\n"
            "Add one with fake.record(messages, cfg, reply) or pass script=[...]."
        )

    def record(self, messages: Messages, cfg: GenerationConfig, reply: str) -> LLMRecord:
        """Add a recording, for example a hand-written reply to a known prompt."""
        rec = make_record(messages, cfg, reply)
        self.records[rec.key] = rec
        return rec

    def save(self, path: str | Path) -> None:
        """Write the recordings as JSONL, sorted by key so diffs stay small."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [self.records[k].model_dump_json() for k in sorted(self.records)]
        path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")

    @property
    def remaining(self) -> int:
        """Scripted replies not used yet; a test can assert it is 0."""
        return len(self.script)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
        """Replace `llm.complete` for the rest of the test. Callers must use `llm.complete(...)`,
        not `from snakepp.generation.llm import complete`, for the swap to reach them."""
        monkeypatch.setattr(llm, "complete", self.complete)
        return self
