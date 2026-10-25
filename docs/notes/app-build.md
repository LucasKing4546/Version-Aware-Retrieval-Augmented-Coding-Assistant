# App build (2026-10-25)

Everything below is on the local branch chain starting at `dev/integration`, which already includes PRs #42–#45.
It has not been pushed yet. `pytest -q`: 87 passed, `ruff check .`: clean.

| Area | What exists now |
|---|---|
| Retrieval | BM25 with an identifier-aware tokenizer · dense (bge-small, FAISS, embeddings cached) · RRF · cross-encoder rerank (bge-reranker-base) · version filter with detection from an explicit target, the query text or the installed packages · `Retriever` driven by the config |
| Generation | `llm.complete` for vLLM / LM Studio / OpenAI / Anthropic, every call cached · no-context and numbered-context prompts · citation parsing |
| Evaluation | sandbox runner (local Python 3.10 environment or Docker, cached) · DS-1000 (887 problems), HumanEval (164), MBPP (257 test + 43 dev) · reference solutions dropped at load |
| Version set | 15 pandas questions × 2 versions, each scored in its own environment; `validate`: 46/46 checks as expected |
| Corpus | docs for 6 libraries × 2 versions (Python: one): 53,047 chunks · Stack Overflow parser + PyPI version inference · decontamination (MinHash + Jaccard + optional cosine) |
| Pipeline | `run` / `sweep` with `--model`, `--gen-workers`; results are never overwritten; configs missing the SO corpus are skipped, not silently run on docs only |
| Demo / analysis | Gradio app (`demo/app.py`) · `analysis/make_tables.py` (pass@1 + bootstrap CI, McNemar vs c0, error types) |
| Infra | `scripts/make_sandbox_env.py`, `scripts/kaggle_run.sh` (vLLM + full sweep), smoke test with 15 checks |

## First numbers (dev runs, Llama-3.1-8B on LM Studio, laptop CPU — not results)
- DS-1000, c0, first 5 problems: pass@1 0.4.
- version_set, c0: pass@1 0.80 (24/30).
- version_set, c1_docs: see the run log. The results go into `data/results/version_set/`.

## Blocked on decisions
See the questions in the chat: Stack Overflow data source, dev split for tuning, version-filter minor gap,
how to integrate the branches, and where the full runs happen.
