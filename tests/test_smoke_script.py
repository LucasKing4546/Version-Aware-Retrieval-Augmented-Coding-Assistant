import importlib.util
import sys
from pathlib import Path


def _script():
    spec = importlib.util.spec_from_file_location("smoke_test", Path("scripts/smoke_test.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_checks_counts_each_status(capsys):
    smoke = _script()

    def skip():
        raise smoke.Skip("not here")

    def fail():
        raise AssertionError("boom")

    counts = smoke.run_checks([("ok", lambda: "fine"), ("skip", skip), ("fail", fail)])
    assert counts == {"PASS": 1, "FAIL": 1, "SKIP": 1}
    out = capsys.readouterr().out
    assert "[PASS] ok" in out and "[SKIP] skip" in out and "AssertionError: boom" in out


def test_fixture_checks_pass_on_main():
    smoke = _script()
    assert "200 unique chunks" in smoke.check_fixture_contents()
    assert smoke.check_fixture_deterministic()
