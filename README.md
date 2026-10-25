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

Full install (retrieval models, LLM clients, demo, analysis):
`pip install -e ".[dev,retrieval,gen,demo,analysis]"`. On a CPU-only machine install torch first with
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Build the data (all under `data/`, git-ignored)

```bash
python scripts/make_sandbox_env.py                    # Python 3.10 + DS-1000 pinned versions (no Docker needed)
python -m snakepp.corpus.docs build-all               # docs chunks: 6 libraries x their versions (~53k chunks)
python scripts/download_so_dump.py                    # Stack Overflow dump (69 GB), then extract Posts.xml
python -m snakepp.corpus.stackoverflow build data/raw/stackoverflow/Posts.xml --limit 50000
python -m snakepp.corpus.decontam run                 # so_clean.parquet + so_leaky.parquet
python -m snakepp.benchmarks.version_set.loader validate   # proves every version-set test
```

## LLM

Free, open models only: set `VLLM_BASE_URL` in `.env` to any OpenAI-compatible server — vLLM on Kaggle/Colab
(`Qwen/Qwen2.5-Coder-7B-Instruct`, the config default) or LM Studio locally (`http://localhost:1234/v1`, then pass
`--model <name shown in LM Studio>`). Every call is cached in `data/cache/llm`.

## Smoke test

```bash
python scripts/smoke_test.py            # pytest, ruff and a functional check of every finished piece
python scripts/smoke_test.py --github   # also checks the issue board and the last CI run on main
```

Prints PASS / FAIL / SKIP per check. SKIP means that piece is not on the current branch yet.

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
python -m snakepp.pipeline run configs/c5_full.yaml --benchmark ds1000 --limit 50
python -m snakepp.pipeline sweep --benchmarks ds1000 --benchmarks humaneval --benchmarks mbpp --benchmarks version_set
python analysis/make_tables.py                        # pass@1 + CI, paired test vs c0_none, error types
python demo/app.py --model meta-llama-3.1-8b-instruct # Gradio demo on http://localhost:7860
```

Benchmarks: `ds1000` (887 problems, 5 libraries), `humaneval` (164), `mbpp` (257, test split 11-510), `version_set`
(15 pandas questions x 2 versions). Results: `data/results/<benchmark>/<config>_<date>[_n].{json,jsonl}`, never
overwritten.

## Plan

See `docs/PLAN.md` for milestones and `docs/tasks.yaml` for every task with its owner and due date.
`python scripts/create_issues.py` turns the task list into GitHub issues (needs the `gh` CLI, logged in).

## Licensing

Stack Overflow content is CC BY-SA; every chunk keeps its author link and license. Each documentation set keeps its own license.
