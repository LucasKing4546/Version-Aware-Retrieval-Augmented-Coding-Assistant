"""Fetch Stack Overflow questions with their accepted answers from the Stack Exchange API (raw JSON only).

Walks forward in creation date instead of deep paging: each request starts at the newest post seen so far.
Every page lands in data/raw/so/<tag>/<fromdate>.json and the cursor in state.json, so a run killed by the
daily quota resumes where it stopped. Turning raw posts into chunks happens in a later step.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.error import HTTPError

from snakepp.ingest.pypi import USER_AGENT
from snakepp.paths import DATA

API = "https://api.stackexchange.com/2.3"
RAW_DIR = DATA / "raw" / "so"


def _gunzip(body: bytes) -> bytes:
    return gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body


def _get(path: str, **params) -> dict:
    params["site"] = "stackoverflow"
    if key := os.environ.get("SE_API_KEY"):
        params["key"] = key
    req = urllib.request.Request(
        f"{API}{path}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(_gunzip(resp.read()))
    except HTTPError as e:
        raise RuntimeError(f"Stack Exchange API {e.code}: {_gunzip(e.read())[:300].decode(errors='replace')}") from e
    if data.get("backoff"):  # the API asks for a pause before the next call to this method
        time.sleep(data["backoff"])
    return data


def fetch_questions(
    tag: str, since: date, min_score: int = 3, min_quota: int = 20, out_dir: Path = RAW_DIR
) -> int:
    """Fetch questions tagged `tag` (accepted answer, score >= min_score) created from `since` on.

    Returns the number of pages written. Stops early, resumably, when the daily quota drops below `min_quota`.
    On resume the saved cursor wins over `since`.
    """
    if not os.environ.get("SE_API_KEY"):
        print("SE_API_KEY not set: quota is 300 requests/day instead of 10,000")
    tag_dir = out_dir / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    state = tag_dir / "state.json"
    if state.exists():
        cursor = json.loads(state.read_text(encoding="utf-8"))
        if cursor.get("done"):
            print(f"{tag}: already complete")
            return 0
        fromdate = cursor["fromdate"]
    else:
        fromdate = int(datetime(since.year, since.month, since.day, tzinfo=UTC).timestamp())

    pages = 0
    while True:
        q = _get(
            "/search/advanced", tagged=tag, accepted="True", sort="creation", order="asc",
            fromdate=fromdate, pagesize=100, filter="withbody",
        )  # fmt: skip
        items = q["items"]
        keep = [i for i in items if i["score"] >= min_score and "accepted_answer_id" in i]
        quota = q["quota_remaining"]
        answers = []
        if keep:
            ids = ";".join(str(i["accepted_answer_id"]) for i in keep)
            a = _get(f"/answers/{ids}", pagesize=100, filter="withbody")
            answers, quota = a["items"], a["quota_remaining"]
        (tag_dir / f"{fromdate}.json").write_text(json.dumps({"questions": keep, "answers": answers}), encoding="utf-8")
        pages += 1

        # Inclusive fromdate re-fetches the last post; the parser dedupes by question_id.
        nxt = max((i["creation_date"] for i in items), default=fromdate)
        fromdate = nxt if nxt > fromdate else fromdate + 1
        done = not q["has_more"]
        state.write_text(json.dumps({"fromdate": fromdate, "done": done}), encoding="utf-8")
        print(f"{tag}: page {pages}, kept {len(keep)}/{len(items)}, quota {quota}")
        if done:
            return pages
        if quota < min_quota:
            print(f"{tag}: stopping, quota {quota} < {min_quota}; rerun tomorrow to resume")
            return pages
