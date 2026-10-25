"""Results tables from data/results — owner: Roberto.

For each (benchmark, config) the newest run is used. Writes data/results/tables/<date>/:
  pass_at_1.md / .csv   pass@1 with a 95% bootstrap interval per config and benchmark,
  vs_baseline.md        paired comparison with c0_none (wins / losses, exact McNemar p-value),
  errors.md             error types of failing tasks per config.
Usage: python analysis/make_tables.py [--baseline c0_none]
"""

from __future__ import annotations

import json
import math
import random
from datetime import date
from pathlib import Path

import pandas as pd
import typer

RESULTS = Path("data/results")


def latest_runs(root: Path = RESULTS) -> dict[tuple[str, str], Path]:
    """(benchmark, config) -> newest .jsonl (names sort by date, then _n)."""
    runs: dict[tuple[str, str], Path] = {}
    for summary in sorted(root.glob("*/*.json"), key=lambda p: (p.stat().st_mtime, p.name)):
        cfg = json.loads(summary.read_text(encoding="utf-8"))["config"]
        runs[(summary.parent.name, cfg)] = summary.with_suffix(".jsonl")
    return runs


def load_outcomes(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["result"]["task_id"]: r["result"] for r in rows}


def bootstrap_ci(passed: list[bool], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not passed:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(passed, k=len(passed))) / len(passed) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def mcnemar_p(wins: int, losses: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def pass_table(runs: dict[tuple[str, str], Path]) -> pd.DataFrame:
    rows = []
    for (bench, cfg), path in sorted(runs.items()):
        passed = [r["passed"] for r in load_outcomes(path).values()]
        lo, hi = bootstrap_ci(passed)
        rows.append({"benchmark": bench, "config": cfg, "n": len(passed), "pass@1": sum(passed) / len(passed),
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def baseline_table(runs: dict[tuple[str, str], Path], baseline: str) -> pd.DataFrame:
    rows = []
    for (bench, cfg), path in sorted(runs.items()):
        if cfg == baseline or (bench, baseline) not in runs:
            continue
        a, b = load_outcomes(runs[(bench, baseline)]), load_outcomes(path)
        common = sorted(a.keys() & b.keys())
        wins = sum(b[t]["passed"] and not a[t]["passed"] for t in common)
        losses = sum(a[t]["passed"] and not b[t]["passed"] for t in common)
        rows.append({"benchmark": bench, "config": cfg, "paired_n": len(common), "wins": wins, "losses": losses,
                     "p_mcnemar": round(mcnemar_p(wins, losses), 4)})
    return pd.DataFrame(rows)


def error_table(runs: dict[tuple[str, str], Path]) -> pd.DataFrame:
    rows = []
    for (bench, cfg), path in sorted(runs.items()):
        for r in load_outcomes(path).values():
            if not r["passed"]:
                rows.append({"benchmark": bench, "config": cfg, "error": r["error_type"] or "unknown"})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).value_counts().rename("n").reset_index()


def main(baseline: str = "c0_none", root: Path = RESULTS) -> None:
    runs = latest_runs(root)
    if not runs:
        raise typer.Exit("no results in data/results yet")
    out = root / "tables" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    table = pass_table(runs)
    table.to_csv(out / "pass_at_1.csv", index=False)
    wide = table.pivot(index="config", columns="benchmark", values="pass@1").round(3)
    (out / "pass_at_1.md").write_text(wide.to_markdown() + "\n\n" + table.round(3).to_markdown(index=False),
                                      encoding="utf-8")
    (out / "vs_baseline.md").write_text(baseline_table(runs, baseline).to_markdown(index=False), encoding="utf-8")
    (out / "errors.md").write_text(error_table(runs).to_markdown(index=False), encoding="utf-8")
    typer.echo(wide.to_markdown())
    typer.echo(f"\nwritten to {out}")


if __name__ == "__main__":
    typer.run(main)
