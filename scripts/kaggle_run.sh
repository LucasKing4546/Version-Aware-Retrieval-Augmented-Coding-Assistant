#!/usr/bin/env bash
# Full experiment run on a free Kaggle notebook (GPU T4 x2, internet on). Paste into a cell as:  !bash kaggle_run.sh
# 1. serves Qwen2.5-Coder-7B (AWQ, fits a T4) with vLLM under the config's model name,
# 2. builds the sandbox env and the docs corpus (or reuses data/ from an attached Kaggle dataset),
# 3. runs every config on every benchmark. Re-running resumes: LLM calls and sandbox runs are cached in data/cache.
# Copy data/results (and data/cache to save GPU time next session) to /kaggle/working before the session ends.
set -euo pipefail

REPO=${REPO:-https://github.com/LucasKing4546/Version-Aware-Retrieval-Augmented-Coding-Assistant.git}
BRANCH=${BRANCH:-main}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}
SERVED_NAME=${SERVED_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}   # = GenerationConfig.model, so configs stay unchanged
BENCHMARKS=${BENCHMARKS:-"ds1000 humaneval mbpp version_set"}
LIMIT=${LIMIT:-}                                              # e.g. LIMIT=50 for a quick pass

cd /kaggle/working
[ -d snakepp ] || git clone -q --branch "$BRANCH" "$REPO" snakepp
cd snakepp
pip install -q -e ".[dev,retrieval,gen]" vllm

python -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SERVED_NAME" \
  --quantization awq --dtype half --max-model-len 8192 --tensor-parallel-size 2 --port 8000 \
  > vllm.log 2>&1 &
export VLLM_BASE_URL=http://localhost:8000/v1
until curl -sf "$VLLM_BASE_URL/models" > /dev/null; do sleep 10; done
echo "vLLM is up"

[ -x data/envs/ds1000/bin/python ] || python scripts/make_sandbox_env.py
ls data/chunks/docs_*.parquet > /dev/null 2>&1 || python -m snakepp.corpus.docs build-all

args=()
for b in $BENCHMARKS; do args+=(--benchmarks "$b"); done
python -m snakepp.pipeline sweep "${args[@]}" ${LIMIT:+--limit "$LIMIT"} --gen-workers 16
python analysis/make_tables.py
