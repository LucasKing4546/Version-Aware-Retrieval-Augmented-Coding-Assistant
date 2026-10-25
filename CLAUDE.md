# Snake++ — conventions for Claude Code

Version-aware retrieval-augmented coding assistant for Python (LLMs course project).
Team: Secara Lucas (data), Kevin Petras (retrieval + integration), Enea Sebastian (generation + evaluation),
Dolha Roberto (benchmarks, analysis, demo, report). The full plan is in `docs/PLAN.md`; tasks are in `docs/tasks.yaml`.

## Code rules
- Python 3.11. Format and lint with `ruff`. Type hints everywhere.
- Data passed between modules uses the pydantic models in `snakepp/schemas.py`.
  **Never change `schemas.py` without a team-approved issue** — four people build against it in parallel.
- Every new function gets a pytest test in `tests/`. Run `pytest -q` and `ruff check .` before saying a task is done.
- Keep functions small and one module per concern. CLI entry points use `typer`.
- Read and write data only under `data/` (git-ignored). Never commit large files, model weights or API keys.
- Anything slow (embedding, LLM calls, sandbox execution) is cached in `data/cache/`, keyed by a hash of its inputs and config
  (use `snakepp.cache`).
- Configs live in `configs/*.yaml` and load through `snakepp.config.load_config`. Do not hard-code experiment settings.

## Evaluation hygiene
- Never read, print or paste DS-1000, HumanEval or MBPP reference solutions while building the pipeline or writing prompts.
- Tune k, penalties and fusion constants only on the dev split, never on reported tasks.
- Final result files in `data/results/` are never overwritten; add a date suffix.

## Workflow
- Branch per task: `feat/<area>-<short-name>`. Small PRs (under ~400 lines). CI must be green.
- Stub modules contain `TODO` notes saying what they must do and who owns them; follow those notes.
