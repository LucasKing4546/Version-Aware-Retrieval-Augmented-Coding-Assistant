"""Raw Stack Overflow pages -> one Chunk per question + accepted answer, version inferred from the answer date.

Week 9 adds: an override when the text names a version, the clean/leaky split, and parquet per inferred version.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from snakepp.ingest.docs import html_blocks
from snakepp.ingest.pypi import version_at
from snakepp.schemas import Chunk, make_chunk_id


def _render(body_html: str) -> str:
    return "\n\n".join(f"```\n{t}\n```" if kind == "code" else t for kind, t in html_blocks(body_html))


def qa_chunk(question: dict, answer: dict, library: str, releases: dict[str, str]) -> Chunk:
    """The answer's date picks the version, since the answer is the code people copy."""
    created = datetime.fromtimestamp(answer["creation_date"], UTC).date()
    url = f"https://stackoverflow.com/questions/{question['question_id']}"
    title = html.unescape(question["title"])
    author = html.unescape(answer.get("owner", {}).get("display_name", ""))  # deleted users have no name
    return Chunk(
        id=make_chunk_id(url, "", "0"),
        text=f"{title}\n\nQuestion:\n{_render(question['body'])}\n\nAccepted answer:\n{_render(answer['body'])}",
        source="stackoverflow",
        library=library,
        version=version_at(releases, created),
        version_inferred=True,
        url=url,
        title=title,
        score=answer["score"],
        created=created,
        license=answer["content_license"],
        author=author or None,
    )


def iter_qa(tag_dir: Path) -> Iterator[tuple[dict, dict]]:
    """(question, accepted answer) pairs from raw fetch pages, oldest first, deduped by question_id."""
    seen: set[int] = set()
    for page in sorted((p for p in tag_dir.glob("*.json") if p.stem.isdigit()), key=lambda p: int(p.stem)):
        data = json.loads(page.read_text(encoding="utf-8"))
        answers = {a["answer_id"]: a for a in data["answers"]}
        for q in data["questions"]:
            a = answers.get(q["accepted_answer_id"])
            if a and q["question_id"] not in seen:
                seen.add(q["question_id"])
                yield q, a
