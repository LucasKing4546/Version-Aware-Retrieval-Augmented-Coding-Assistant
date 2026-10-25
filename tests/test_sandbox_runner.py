import sys

import pytest

from snakepp import cache
from snakepp.evaluation.sandbox import runner


@pytest.fixture(autouse=True)
def _use_this_python(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("SNAKEPP_SANDBOX_PYTHON", sys.executable)
    monkeypatch.setenv("SNAKEPP_SANDBOX", "local")


def test_error_type_parsing():
    tb = "Traceback (most recent call last):\n  File x\nAttributeError: 'DataFrame' object has no attribute 'append'"
    assert runner.error_type(tb, 1) == "AttributeError"
    assert runner.error_type("numpy.core._exceptions._UFuncNoLoopError: boom", 1) == "_UFuncNoLoopError"
    assert runner.error_type("", 3) == "ExitCode3"
    assert runner.error_type("whatever", 0) is None


def test_run_program_pass_and_fail():
    assert runner.run_program("assert 1 + 1 == 2").passed
    bad = runner.run_program("import math\nmath.nope()")
    assert not bad.passed and bad.error_type == "AttributeError" and "nope" in bad.stderr_tail


def test_run_program_timeout():
    out = runner.run_program("while True: pass", timeout=1)
    assert out == runner.Outcome(False, "Timeout", "timed out after 1 s")


def test_run_many_keeps_order():
    outs = runner.run_many(["pass", "raise ValueError('x')", "pass"], workers=3)
    assert [o.passed for o in outs] == [True, False, True]
    assert outs[1].error_type == "ValueError"


def test_missing_env_message(monkeypatch, tmp_path):
    monkeypatch.delenv("SNAKEPP_SANDBOX_PYTHON")
    monkeypatch.setattr(runner, "DEFAULT_ENV", tmp_path / "none")
    with pytest.raises(RuntimeError, match="make_sandbox_env"):
        runner.sandbox_python()


def test_run_cached(tmp_path):
    program = f"open(r'{tmp_path / 'ran.txt'}', 'a').write('x')"
    assert runner.run_cached(program).passed and runner.run_cached(program).passed
    assert (tmp_path / "ran.txt").read_text() == "x"  # second call came from the cache
