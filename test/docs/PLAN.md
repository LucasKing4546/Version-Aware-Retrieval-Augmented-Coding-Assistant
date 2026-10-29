# Snake++ fast-track plan (summary)

The full plan with day-by-day tasks and Claude Code prompts is the PDF `Snakepp_Work_Plan.pdf` (keep a copy in `docs/`).
Every task with its owner and due date is in `docs/tasks.yaml`.

## Milestones
| Gate | Date | Must be true |
|---|---|---|
| M0 | Mon 26 Oct | Repo, CI, schemas frozen; SO dump downloading; DS-1000 versions known; generator chosen |
| M1 | Thu 29 Oct | No-retrieval DS-1000 pandas baseline scored; pandas docs chunked; BM25 works |
| M2 | Sun 1 Nov | Hybrid retrieval → generation → tests runs end to end on the pandas slice |
| M3 | Sun 8 Nov | All six configs run on DS-1000 (5 libs) + HumanEval; first results table |
| M4 | Sun 15 Nov | Final numbers frozen, plots, failure analysis, demo recorded |
| M5 | Sun 22 Nov | Report and slides done; v1.0 tagged with a reproduction README |

## Owners
- **Lucas** — `snakepp/corpus/`: docs + Stack Overflow ingestion, version tags, decontamination, licensing
- **Kevin** — repo, `schemas.py`, `snakepp/retrieval/`, `pipeline.py`, configs, integration
- **Enea** — `snakepp/generation/`, `snakepp/evaluation/`: sandbox, harnesses, prompts, citations, all runs
- **Roberto** — `snakepp/benchmarks/version_set/`, `analysis/`, `demo/`, `report/`

## Experiment configurations
| Config | Retrieval | Corpus |
|---|---|---|
| c0_none | none | none |
| c1_docs | hybrid + rerank | docs |
| c2_so | hybrid + rerank | Stack Overflow (clean) |
| c3_dense | dense only | docs + SO |
| c4_hybrid | BM25 + dense fused | docs + SO |
| c5_full | hybrid + rerank | docs + SO |
| c5_full_vf | c5 + soft version filter | docs + SO |
| c5_leaky | c5 on leaky SO index | docs + leaky SO |

## Scope-cut order (if late)
1. Hugging Face extension 2. MBPP 3. second embedding model 4. SciPy 5. SO down to 50k posts.
Never cut: baseline, hybrid + rerank, version filter, decontamination.
