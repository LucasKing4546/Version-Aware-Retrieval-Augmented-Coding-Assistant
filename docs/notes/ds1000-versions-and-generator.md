# Tasks: DS-1000 pinned versions (#4) · Generator choice and timing (#5)

Owner: Enea (done by Kevin with Claude Code) · Milestone: M0 · Due: 2026-10-26 · Branch: `feat/eval-ds1000-versions`

## DS-1000 pinned versions (#4)
The source is `environment.yml` in [xlang-ai/DS-1000](https://github.com/xlang-ai/DS-1000) (environment `ds1000-3.10`, Python 3.10):

| Library | Version |
|---|---|
| pandas | 1.5.3 |
| numpy | 1.26.4 |
| matplotlib | 3.8.4 |
| scikit-learn | 1.4.0 |
| scipy | 1.12.0 |

The same file also pins packages we don't use: pytorch 2.2.0, tensorflow-cpu 2.16.1, seaborn 0.13.2, statsmodels 0.14.1, xgboost 2.0.3 and gensim 4.3.2.

- **`snakepp/evaluation/sandbox/versions.txt`** now holds these five pins, so the Docker sandbox installs exactly them.
  The Dockerfile already uses Python 3.10.
- **`snakepp/evaluation/ds1000.py`** gains two helpers:
  - `pinned_versions()` reads the pins from that file, so it stays the only place they're written.
  - `target_versions("Sklearn")` returns `{"scikit-learn": "1.4.0"}`, the `Task.target_versions` value for a DS-1000 problem.
- **Tests:** `tests/test_ds1000_versions.py` has 3.
- **What this means for the corpus (Lucas):** the DS-1000 target is **pandas 1.5.3**, not 2.x.
  - Index pandas docs at 1.5.3 and at 2.2.x.
  - For DS-1000, the version filter should penalise the 2.x chunks.
  - For the other libraries, index the versions in the table above.
- **Dataset size:** 887 of the 1000 problems use our five libraries: Pandas 291, Numpy 220, Matplotlib 155, Sklearn 115, Scipy 106.
  PyTorch (68) and TensorFlow (45) are excluded.

## Generator choice (#5)
**Decision: open model with vLLM only, no paid APIs.**
- **Model:** `Qwen/Qwen2.5-Coder-7B-Instruct`, the default in `GenerationConfig` (backend `vllm`, temperature 0, max_tokens 1024).
- **Where it runs:** free Kaggle GPUs (2× T4, about 30 h/week) or free Colab (T4).
  The full-precision model barely fits on one T4. Use `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`, or split the full model across both T4s with `--tensor-parallel-size 2`.
- **How the code connects:** set `VLLM_BASE_URL` in `.env`. The `openai` client talks to vLLM with any placeholder key.
  The `anthropic` and `openai` backends stay optional, and nobody needs their keys.

### Timing one DS-1000 problem
I sent one DS-1000 Pandas prompt (problem 0) to a local OpenAI-compatible server (LM Studio).
- **Hardware:** Kevin's laptop, an Intel Core Ultra 7 155H with Arc integrated graphics and no NVIDIA GPU.
- **Model:** Llama-3.1-8B-Instruct as a GGUF file, because Qwen2.5-Coder isn't downloaded there.
- **Settings:** temperature 0. I logged timings only and printed no reference solution.

| Run | Latency | Prompt tokens | Output tokens | Speed |
|---|---|---|---|---|
| Model not loaded yet | 28.6 s | 477 | 128 | 4.5 tok/s |
| Model already loaded | 15.3 s | 477 | 163 | 10.7 tok/s |

At about 15–20 s per problem, 887 problems × 8 configs = 7,096 generations would take **30–40 hours of nonstop computing on a laptop**.
Retrieval configs make the prompts 2–3k tokens longer, so it would be slower still.
Laptops are fine for development and the demo. Full runs go on Kaggle with vLLM, which runs many prompts at once and should be much faster; that still needs measuring.
Every call is cached in `data/cache/`, so re-running costs nothing.

**Still open:** time the same problem with Qwen2.5-Coder-7B on a Kaggle T4 with vLLM, then check that the answer passes the official test inside the sandbox.
That check needs `sandbox/runner.py` (due M1).
