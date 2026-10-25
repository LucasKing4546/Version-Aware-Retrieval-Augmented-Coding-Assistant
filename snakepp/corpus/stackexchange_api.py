"""Stack Overflow subset through the public Stack Exchange API — owner: Lucas.

Alternative to the 69 GB dump: for each library, the highest-voted python questions with an accepted answer, year
by year (so posts span many library versions), plus their accepted answers. Every response is cached in
data/cache/seapi, so a run interrupted by the daily quota (300 requests without a key, 10,000 with SE_API_KEY in
.env) resumes for free. Items become Posts.xml-style rows and reuse stackoverflow.make_chunk.
Usage: python -m snakepp.corpus.stackexchange_api build --per-library 2000
"""

from __future__ import annotations

import gzip
import html
import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import load_dotenv

from snakepp import cache
from snakepp.corpus.io import CHUNKS_DIR, write_chunks
from snakepp.corpus.stackoverflow import make_chunk

API = "https://api.stackexchange.com/2.3"
TAGS = {"pandas": "python;pandas", "numpy": "python;numpy", "matplotlib": "python;matplotlib",
        "scikit-learn": "python;scikit-learn", "scipy": "python;scipy"}
YEARS = range(2012, 2026)

app = typer.Typer(help="Stack Overflow subset from the Stack Exchange API")


@app.callback()
def _main() -> None:
    """Keep subcommand names on the command line."""


class QuotaExceeded(RuntimeError):
    pass


def _http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = resp.read()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def get(endpoint: str, params: dict, fetch=_http_get) -> dict:
    """One API call, cached by endpoint + params (the key is not part of the cache key)."""
    def compute() -> dict:
        load_dotenv()
        query = {**params, "site": "stackoverflow", **({"key": k} if (k := os.environ.get("SE_API_KEY")) else {})}
        data = fetch(f"{API}/{endpoint}?{urllib.parse.urlencode(query)}")
        if "error_id" in data:
            raise QuotaExceeded(f"{data.get('error_name')}: {data.get('error_message')} "
                                "(set SE_API_KEY in .env or rerun tomorrow; finished calls are cached)")
        time.sleep(data.get("backoff", 0))
        return data

    return cache.cached_json("seapi", cache.key(endpoint, params), compute)


def question_row(item: dict) -> dict:
    return {"Id": str(item["question_id"]), "Title": html.unescape(item["title"]), "Body": item.get("body", ""),
            "Score": str(item["score"]), "Tags": "|" + "|".join(item.get("tags", [])) + "|",
            "AcceptedAnswerId": str(item.get("accepted_answer_id", "")),
            "CreationDate": datetime.fromtimestamp(item["creation_date"], UTC).isoformat()[:19]}


def answer_row(item: dict) -> dict:
    owner = item.get("owner", {})
    return {"Id": str(item["answer_id"]), "Body": item.get("body", ""), "Score": str(item["score"]),
            "OwnerUserId": str(owner.get("user_id", "")),
            "OwnerDisplayName": html.unescape(owner.get("display_name", "")),
            "ContentLicense": item.get("content_license", ""),
            "CreationDate": datetime.fromtimestamp(item["creation_date"], UTC).isoformat()[:19]}


def top_questions(library: str, year: int, n: int, min_score: int, fetch=_http_get) -> list[dict]:
    """Up to n highest-voted questions of that year with an accepted answer and score >= min_score."""
    start = int(datetime(year, 1, 1, tzinfo=UTC).timestamp())
    end = int(datetime(year + 1, 1, 1, tzinfo=UTC).timestamp())
    out: list[dict] = []
    page = 1
    while len(out) < n:
        data = get("search/advanced", {"tagged": TAGS[library], "accepted": "True", "sort": "votes", "order": "desc",
                                       "fromdate": start, "todate": end, "page": page, "pagesize": 100,
                                       "filter": "withbody"}, fetch)
        items = [i for i in data.get("items", []) if i.get("accepted_answer_id")]
        out += [i for i in items if i["score"] >= min_score]
        if not data.get("has_more") or (items and items[-1]["score"] < min_score):
            break
        page += 1
    return out[:n]


def accepted_answers(answer_ids: list[int], fetch=_http_get) -> dict[int, dict]:
    found = {}
    for i in range(0, len(answer_ids), 100):
        batch = ";".join(str(a) for a in answer_ids[i : i + 100])
        data = get(f"answers/{batch}", {"pagesize": 100, "filter": "withbody"}, fetch)
        found.update({a["answer_id"]: a for a in data.get("items", [])})
    return found


def iter_pairs(library: str, per_library: int, min_score: int = 3, fetch=_http_get) -> Iterator[tuple[dict, dict]]:
    per_year = max(1, per_library // len(YEARS))
    for year in YEARS:
        questions = top_questions(library, year, per_year, min_score, fetch)
        answers = accepted_answers([q["accepted_answer_id"] for q in questions], fetch)
        for q in questions:
            if (a := answers.get(q["accepted_answer_id"])) is not None:
                yield question_row(q), answer_row(a)


@app.command()
def build(per_library: int = 2000, min_score: int = 3, out: Path = CHUNKS_DIR / "so_all.parquet") -> None:
    """Write so_all.parquet (then run decontam). ~2 * per_library / 100 requests per library."""
    from snakepp.corpus.versions import infer_version

    chunks: dict[str, object] = {}  # a post tagged with two libraries is kept once, under the first
    for library in TAGS:
        before = len(chunks)
        for q, a in iter_pairs(library, per_library, min_score):
            chunk = make_chunk(q, a, library, infer_version)
            chunks.setdefault(chunk.id, chunk)
        typer.echo(f"{library}: {len(chunks) - before} new Q&A pairs")
    typer.echo(f"{out}: {write_chunks(chunks.values(), out)} chunks")


if __name__ == "__main__":
    app()
