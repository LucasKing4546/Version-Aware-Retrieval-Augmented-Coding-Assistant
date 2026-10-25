# App build (2026-10-25)

Everything below is on the local branch chain starting at `dev/integration`, which already includes PRs #42–#45.
It is pushed as one combined PR (team decision). `pytest -q`: 94 passed, `ruff check .`: clean.

| Area | What exists now |
|---|---|
| Retrieval | BM25 with an identifier-aware tokenizer · dense (bge-small, FAISS, embeddings cached) · RRF · cross-encoder rerank (bge-reranker-base) · version filter with detection from an explicit target, the query text or the installed packages · `Retriever` driven by the config |
| Generation | `llm.complete` for vLLM / LM Studio / OpenAI / Anthropic, every call cached · no-context and numbered-context prompts · citation parsing |
| Evaluation | sandbox runner (local Python 3.10 environment or Docker, cached) · DS-1000: 786 test + 101 dev (10% of problem groups per library, perturbations kept together) · HumanEval (164) · MBPP (257 test + 43 dev) · reference solutions dropped at load |
| Version set | 15 pandas questions × 2 versions, each scored in its own environment; `validate`: 46/46 checks as expected |
| Corpus | docs for 6 libraries × 2 versions (Python: one): 53,047 chunks · Stack Overflow: 7,829 Q&A pairs from the Stack Exchange API (2012–2025, 252 requests) plus a dump parser · PyPI version inference · decontamination: 57 DS-1000 matches → so_clean 7,796, so_leaky 7,829 |
| Pipeline | `run` / `sweep` with `--model`, `--gen-workers`, `--set field=value` (tuning); results are never overwritten; configs missing the SO corpus are skipped, not silently run on docs only |
| Demo / analysis | Gradio app (`demo/app.py`) · `analysis/make_tables.py` (pass@1 + bootstrap CI, McNemar vs c0, error types) |
| Infra | `scripts/make_sandbox_env.py`, `scripts/kaggle_run.sh` (vLLM + full sweep), smoke test with 15 checks |

## First numbers (dev runs, Llama-3.1-8B on LM Studio, laptop CPU — not results)
- DS-1000, c0, first 5 problems: pass@1 0.4.
- version_set, c0: pass@1 0.80 (24/30).
- version_set, c1_docs: see the run log. The results go into `data/results/version_set/`.

## Decisions (2026-10-25)
- Stack Overflow from the Stack Exchange API, not the 69 GB dump. Without a key that's 300 requests a day; set `SE_API_KEY` for more.
- Tune only on `ds1000_dev` + `mbpp_dev`. Report on `ds1000`, `humaneval`, `mbpp`, `version_set`.
- Full runs on Kaggle with `scripts/kaggle_run.sh`. Upload `data/chunks` as the Kaggle dataset `snakepp-data`.
