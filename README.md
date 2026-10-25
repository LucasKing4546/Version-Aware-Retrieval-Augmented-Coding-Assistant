# Snake++

A version-aware retrieval-augmented coding assistant for Python. It answers programming questions with code grounded in
library documentation and Stack Overflow answers, filters retrieved context by the library versions the user has installed,
cites its sources, and is evaluated with unit tests (DS-1000, HumanEval, MBPP and our own version-sensitive set).

Team: Secara Lucas · Kevin Petras · Enea Sebastian · Dolha Roberto

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add API keys here, never commit .env
pytest -q && ruff check .
```

Optional heavy extras: `pip install -e ".[retrieval]"` (sentence-transformers, FAISS) and `pip install -e ".[gen]"` (LLM clients).

## Layout

| Path | Owner | Purpose |
|---|---|---|
| `snakepp/schemas.py` | all | Frozen interfaces between modules |
| `snakepp/corpus/` | Lucas | Docs and Stack Overflow ingestion, version tags, decontamination |
| `snakepp/retrieval/` | Kevin | BM25, dense, fusion, reranking, version filter |
| `snakepp/generation/` | Enea | LLM wrapper, prompts, citation parsing |
| `snakepp/evaluation/` | Enea | Sandbox and benchmark harnesses, metrics |
| `snakepp/benchmarks/version_set/` | Roberto | Version-sensitive questions and tests |
| `snakepp/pipeline.py` | Kevin | config → retrieve → generate → evaluate |
| `analysis/`, `demo/`, `report/` | Roberto | Notebooks, Gradio demo, report and slides |
| `configs/` | Kevin | One YAML per experiment configuration |
| `data/` | — | Git-ignored: raw, chunks, indexes, cache, results |

## Running experiments

```bash
python -m snakepp.pipeline run --config configs/c5_full.yaml --benchmark ds1000 --limit 50
python -m snakepp.pipeline sweep --configs "configs/*.yaml" --benchmarks ds1000 humaneval mbpp version_set
```

## Plan

See `docs/PLAN.md` for milestones and `docs/tasks.yaml` for every task with its owner and due date.
`python scripts/create_issues.py` turns the task list into GitHub issues (needs the `gh` CLI, logged in).

## Licensing

Stack Overflow content is CC BY-SA; every chunk keeps its author link and license. Each documentation set keeps its own license.
