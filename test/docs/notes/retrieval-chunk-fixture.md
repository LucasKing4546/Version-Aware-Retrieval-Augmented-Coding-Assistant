# Task: Stub chunk fixture (200 chunks) for retrieval tests

Owner: Kevin · Milestone: M1 · Due: 2026-10-27 · Branch: `feat/retrieval-chunk-fixture`

## What was done
- **`scripts/make_chunk_fixture.py`** builds the fixture with a deterministic generator (a typer CLI).
  Run `python scripts/make_chunk_fixture.py` to rebuild it.
- **`tests/fixtures/chunks_200.jsonl`** (~88 KB) holds 200 `Chunk` objects, one JSON object per line.
  - **Sources:** 130 docs chunks and 70 Stack Overflow chunks.
  - **Libraries:** pandas, numpy, matplotlib, scikit-learn, scipy and the Python standard library, two versions each.
  - **API changes between versions:** `DataFrame.append`, `iteritems`, `applymap`, `np.product`, `np.in1d`, `cm.get_cmap`,
    the `OneHotEncoder` `sparse` parameter, `simps`, `datetime.utcnow`.
  - **Stack Overflow chunks** have an author, the CC BY-SA 4.0 license, a date, a score and an inferred version.
    Some have `version=None` to stand for posts where version inference failed.
  - **Source of the text:** we wrote it ourselves from general API knowledge. Nothing comes from DS-1000, HumanEval or MBPP.
- **`tests/conftest.py`** adds pytest fixtures shared by all tests:
  - `chunks_200` returns the 200 chunks as a `list[Chunk]`.
  - `chunks_by_id` returns them as a `dict[str, Chunk]`, which is the input `version_filter.apply` expects.
  - `chunks_parquet` writes the chunks to a temporary parquet file, the format BM25 and dense will index.
- **`tests/test_chunk_fixture.py`** checks five things:
  - the committed file still matches the generator's output;
  - the JSONL file reads back unchanged;
  - the size and spread: 200 unique ids, both sources, all six libraries;
  - every Stack Overflow chunk keeps its author, license and date;
  - the version filter gets both compatible and incompatible chunks, and the parquet file reads back correctly.

## Notes
- The fixture is JSONL rather than parquet because `.gitignore` excludes `*.parquet`. Tests that need parquet use `chunks_parquet`.
- Changing the fixture means editing `APIS` in the script and rerunning it. Otherwise `test_fixture_matches_generator` fails.
- `pytest -q`: 13 passed. `ruff check .`: clean.
