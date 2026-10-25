# Running the full experiments on Kaggle (free)

You need a Kaggle account with a verified phone number, which GPU notebooks require. The whole run is free.

## 1. Upload the corpus (once)
1. Get `snakepp-data.zip` (about 47 MB) from Kevin. It contains `chunks/` with the docs files plus `so_clean` and `so_leaky`.
2. On kaggle.com go to **Datasets → New Dataset**, upload the zip and name the dataset **`snakepp-data`**. Keep it private.

## 2. Create the notebook
1. Go to **Code → New Notebook**.
2. In the right panel under **Session options**, set:
   - **Accelerator:** GPU T4 x2
   - **Internet:** On
   - **Persistence:** Files only, so `data/cache` survives restarts
3. Under **Input → Add Input**, add your `snakepp-data` dataset.

## 3. Run (one cell)
```bash
!git clone -q https://github.com/LucasKing4546/Version-Aware-Retrieval-Augmented-Coding-Assistant.git snakepp
!bash snakepp/scripts/kaggle_run.sh
```
- **Quick test first:** `!LIMIT=20 bash snakepp/scripts/kaggle_run.sh` runs 20 tasks per benchmark and config.
- **What the script does:**
  1. installs everything;
  2. starts vLLM with Qwen2.5-Coder-7B (AWQ);
  3. builds the Python 3.10 test environment;
  4. runs every config on DS-1000, HumanEval, MBPP and the version set;
  5. writes the tables.
- **If the session stops** (12 h limit, or ~30 GPU hours per week): run the same cell again. Every model call and test run is cached, so it continues where it stopped.

## 4. Save the results
```bash
!cd /kaggle/working/snakepp && zip -qr /kaggle/working/results.zip data/results
```
Download `results.zip` from the **Output** panel and send it to Kevin. Never overwrite an existing results file; new runs get a date suffix automatically.

## Tuning (only on the dev splits)
```bash
!cd /kaggle/working/snakepp && python -m snakepp.pipeline run configs/c5_full_vf.yaml --benchmark ds1000_dev --set version_filter.penalty=0.5 --gen-workers 16
```
Use only `ds1000_dev` and `mbpp_dev` for tuning. Put the chosen values into `configs/*.yaml` before the final sweep.
