# Kevin — handoff log

Lane: **Retrieval + MCP** (indexer, RRF, reranker, version resolver/filter, retrieval metrics, MCP server, web demo).
Experiments: E2, E4, E5, E9 latency. Failure tags: retrieval misses, wrong-version chunks.
Source of truth: "Snake++ — parallel work split for 4 developers" (Lucas, 15 Nov), plus ARCHITECTURE.md.

Update the **Where I left off** section at the end of every session.

---

## Where I left off (15 Nov 2026)

**Task 1: Contracts PR (due Mon 16 Nov, merge the same evening).** Status: pushed to `origin/Kevin`; PR #48 is open. The CI workflow was removed on 15 Nov (it failed on lint errors
in the old `test/` folder). There is no CI now, so run pytest and ruff locally before pushing. Tell the team at the
meeting, because the plan's repo rules ask for CI.

Done:
- [x] `snakepp/schemas.py`: Chunk, Retrieved, Task, Generation, EvalResult, plus the new **RunResult**, plus gold types
      (GoldChunkRef, G1Item, G2Item, G3Item).
- [x] Signature stubs that raise `NotImplementedError`, one owner each:
  - Lucas: `ingest/cli.py` (`docs`, `so`), `ingest/chunks.py::load_chunks`
  - Kevin: `retrieval/retriever.py` (`Retriever.from_config().retrieve()`, `get_chunk`), `retrieval/versions.py`
    (`detect_versions`, `list_indexed_versions`)
  - Enea: `generation/llm.py::complete`, `generation/tools.py::Tools` (the 6 MCP tools), `pipeline.py::answer`, `agent.py::solve`
  - Roberto: `evaluation/tasks.py::load_tasks`, `evaluation/sandbox.py::run`, `evaluation/scorer.py::score`
- [x] `snakepp/config.py` with the contract keys: retrieval `mode`, `sources`, `k`, `rerank`, `corpus` (clean/leaky),
      `version_filter.mode` and `penalty`, `generation.model`.
- [x] `configs/`: c0_none, c1_docs, c2_so, c3_dense, c4_hybrid, c5_full, c5_soft_01/03/05, c5_hard, c5_leaky.
- [x] `configs/pins.yaml`: ds1000 env, plus pandas_old 1.5.3 and pandas_new 2.2.2 for the version trap.
- [x] `snakepp/paths.py`: folder layout helpers (`chunk_path`, `results_path`), `gold/` templates, `tests/fakes/`.
- [x] Tests: `tests/test_contracts.py` and the existing fixture tests. 39 pass; `ruff check snakepp tests scripts` is clean.

To do before or at the contract meeting:
- [ ] **Delete the old `test/` folder** (the old prototype). The useful files were already moved to the root. The old code
      is still in git history: `git show 1b3519d:test/snakepp/retrieval/bm25.py` (also dense.py, fusion.py, rerank.py,
      version_filter.py). These are a good starting point for BM25, dense and RRF.
- [ ] Confirm the **DS-1000 pins** in `configs/pins.yaml` (marked TODO) against DS-1000's official environment.
- [ ] Get the team to agree on the decisions I added beyond ARCHITECTURE.md §4:
      `Retrieved.chunk` (optional, filled so callers don't need `get_chunk`), `Chunk.heading_path` (needed for gold keys),
      `Task.split` and `Task.sample`, `Generation.tool_calls` and `corpus_hash`, and benchmark names `ds1000|humaneval|mbpp|g1|mini`.
- [ ] The branch is named `Kevin` (as I asked). The working rules want `feat/<lane>-<short-name>`, so maybe rename it to
      `feat/retrieval-contracts` before opening the PR.
- [ ] Commit, push, open the "contracts" PR, and get all four to approve.

## Next tasks (from the plan)

| When | Task |
|---|---|
| Tue 17 Nov | `tests/fakes/fake_retriever.py`: **FakeRetriever**, keyword match over `tests/fixtures/chunks_200.jsonl`, implements `retrieve()` |
| Week 8 (to 22 Nov) | BM25 with an identifier-aware tokenizer · bge-small + FAISS · RRF · `retrieve()` on the fixture |
| Week 9 (to 29 Nov) | Indexer on real parquet · cross-encoder reranker · version resolver, soft and hard filter · recall@k, MRR, version-correct@k · MCP server with search_docs, search_stackoverflow, get_chunk |
| **Gate 2, Sun 29 Nov** | c0–c5 run on dev (with Enea) |
| **Wed 2 Dec** | `trace` command (BM25, dense, fused and reranked lists, version decision, what was dropped) on the DataFrame.append question, plus the written **conflict policy** · MCP server with all 6 tools handed to Enea |
| Week 10 (to 6 Dec) | detect_versions, list_indexed_versions, run_python tools · manifest resource + prompt · stdio + HTTP · connect to Claude Code |
| Gate 3, Sun 6 Dec | Agent answers a G1 task through MCP |
| Week 11 | E4 (optional) · E9 latency per tool · web demo over MCP HTTP |
| Week 12 | Tag failure cases · tag `v0.9` |
| Week 13 | 2-minute demo video |

## Inputs I'm waiting on

| From | What | When | Until then, use |
|---|---|---|---|
| Lucas | pandas parquet, both versions | Fri 20 Nov | fixture |
| Roberto | LocalRunner (behind `run_python`) | Wed 18 Nov | hand-written snippets |
| Lucas | all libraries, clean and leaky SO | Thu 26 Nov | pandas only |
| Roberto | Docker `sandbox.run()` | Fri 27 Nov | LocalRunner |
| Roberto, Lucas | G1 gold chunks, G2 draft queries | Sun 29 Nov | metrics on the fixture |

## Local setup notes

- Venv: `.venv` (Python 3.12). Run `.venv/Scripts/python.exe -m pytest -q` and `.venv/Scripts/ruff.exe check snakepp tests scripts`.
- `data/` (~2.4 GB of old prototype caches: LLM, sandbox, Stack Exchange API) is git-ignored. I kept it because the
  `data/cache/seapi` responses may save Lucas's API quota. Ask him before deleting it.
