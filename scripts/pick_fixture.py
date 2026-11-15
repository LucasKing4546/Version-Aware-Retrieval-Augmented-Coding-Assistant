"""Pick the real-data fixture tests/fixtures/fixture_chunks.jsonl from the ingested pandas corpus.

Usage (repo root, after the pandas docs and Stack Overflow fetches):
    python -m scripts.pick_fixture        # or: pip install -e . && python scripts/pick_fixture.py

About 200 real chunks: pandas 1.5.3 and 2.2.3 docs for APIs that changed across 2.0 (DataFrame.append,
iteritems, applymap -> map, lookup, mad, is_monotonic) plus nearby distractors, and a few Stack Overflow answers
that use the old APIs, with versions inferred from their dates. A page missing from one version is skipped:
that absence is the version trap. Re-running on the same raw data gives a byte-identical file.
"""

from __future__ import annotations

import sys
from pathlib import Path

from snakepp.ingest.chunker import chunk_sections
from snakepp.ingest.docs import DOCS_BASE, DOCS_LICENSE, RAW_DIR, inventory_pages, parse_page
from snakepp.ingest.pypi import load_releases
from snakepp.ingest.so_parse import iter_qa, qa_chunk
from snakepp.paths import DATA
from snakepp.schemas import Chunk

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "fixture_chunks.jsonl"
VERSIONS = ("1.5.3", "2.2.3")
API = [
    # changed or removed in 2.0, or renamed in 2.1
    "DataFrame.append", "Series.append", "concat", "DataFrame.iteritems", "Series.iteritems", "DataFrame.items",
    "Series.items", "DataFrame.applymap", "DataFrame.map", "DataFrame.lookup", "DataFrame.mad",
    "Index.is_monotonic", "Index.is_monotonic_increasing", "DataFrame.swapaxes", "Series.str.replace",
    # distractors: common APIs a retriever must not confuse with the above
    "DataFrame.groupby", "DataFrame.merge", "DataFrame.loc", "DataFrame.fillna", "DataFrame.ffill", "read_csv",
    "DataFrame.pivot_table", "DataFrame.drop", "DataFrame.value_counts", "DataFrame.copy", "Series.view",
    "DataFrame.explode", "to_datetime",
]  # fmt: skip
# page -> keep only sections whose heading path contains this text (None keeps the whole page)
PAGES: dict[str, str | None] = {f"reference/api/pandas.{name}.html": None for name in API} | {
    "user_guide/merging.html": None,
    "whatsnew/v2.0.0.html": "Removal of prior version deprecations",
}
SO_QUESTIONS = [
    39815646, 36526282, 57000903, 39998262, 41234822, 37009287, 53924656, 51774826,  # DataFrame/Series.append
    38387529,  # iteritems
    43654727, 51272815,  # applymap
]  # fmt: skip


def docs_chunks(version: str) -> list[Chunk]:
    root = RAW_DIR / "pandas" / version
    listed = set(inventory_pages((root / "objects.inv").read_bytes()))
    out: list[Chunk] = []
    for page, heading in PAGES.items():
        if page not in listed:
            continue  # the API doesn't exist in this version
        if not (root / page).exists():
            sys.exit(f"missing {version}/{page}: finish `ingest docs --library pandas --version {version}` first")
        chunks = chunk_sections(
            parse_page((root / page).read_bytes()),
            url=DOCS_BASE["pandas"].format(version=version) + page,
            library="pandas", version=version, license=DOCS_LICENSE["pandas"],
        )  # fmt: skip
        out += [c for c in chunks if heading is None or heading in c.heading_path]
    return out


def so_chunks() -> list[Chunk]:
    releases = load_releases("pandas")
    found = {q["question_id"]: qa_chunk(q, a, "pandas", releases) for q, a in iter_qa(DATA / "raw" / "so" / "pandas")}
    if missing := [qid for qid in SO_QUESTIONS if qid not in found]:
        sys.exit(f"Stack Overflow questions not fetched yet: {missing}")
    return [found[qid] for qid in SO_QUESTIONS]


def main() -> None:
    chunks = [c for v in VERSIONS for c in docs_chunks(v)] + so_chunks()
    assert len({c.id for c in chunks}) == len(chunks), "duplicate chunk ids"
    OUT.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
    by = {}
    for c in chunks:
        by[(c.source, c.version)] = by.get((c.source, c.version), 0) + 1
    print(f"{len(chunks)} chunks -> {OUT}")
    for key, n in sorted(by.items(), key=str):
        print(f"  {key}: {n}")


if __name__ == "__main__":
    main()
