import importlib.util
import json
import sys
from pathlib import Path


def _mod():
    spec = importlib.util.spec_from_file_location("make_tables", Path("analysis/make_tables.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run(root, bench, cfg, passed):
    d = root / bench
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"generation": {}, "result": {"task_id": f"t{i}", "passed": p, "error_type": None if p else "TypeError"}}
            for i, p in enumerate(passed)]
    (d / f"{cfg}_2026-10-25.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / f"{cfg}_2026-10-25.json").write_text(json.dumps({"config": cfg}), encoding="utf-8")


def test_mcnemar_and_bootstrap():
    m = _mod()
    assert m.mcnemar_p(0, 0) == 1.0
    assert abs(m.mcnemar_p(9, 1) - 0.0215) < 1e-3
    lo, hi = m.bootstrap_ci([True] * 5 + [False] * 5)
    assert 0.1 <= lo <= 0.5 <= hi <= 0.9


def test_tables(tmp_path):
    m = _mod()
    _write_run(tmp_path, "ds1000", "c0_none", [True, False, False, False])
    _write_run(tmp_path, "ds1000", "c5_full", [True, True, True, False])
    runs = m.latest_runs(tmp_path)
    table = m.pass_table(runs).set_index("config")
    assert table.loc["c5_full", "pass@1"] == 0.75
    vs = m.baseline_table(runs, "c0_none").iloc[0]
    assert (vs["wins"], vs["losses"]) == (2, 0)
    assert set(m.error_table(runs)["error"]) == {"TypeError"}
