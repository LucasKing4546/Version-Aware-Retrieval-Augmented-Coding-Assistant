"""Stack Overflow corpus — owner: Lucas.

Streams Posts.xml with lxml iterparse (flat memory) and keeps python questions tagged with one of our libraries
that have an accepted answer and score >= min_score. One Chunk per question + accepted answer: HTML becomes text
with <pre><code> as fenced code, and each chunk keeps its URL, score, date, author and CC BY-SA license.
Usage: python -m snakepp.corpus.stackoverflow build data/raw/stackoverflow/Posts.xml --limit 20000
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import date, datetime
from pathlib import Path

import lxml.html
import typer
from lxml import etree

from snakepp.corpus.io import CHUNKS_DIR, write_chunks
from snakepp.schemas import Chunk, make_chunk_id

# Fabricated rows Stack Exchange planted in the 2025-06-30 dump ("v1" watermark), see
# https://github.com/LunarWatcher/se-data-dump-transformer/blob/master/docs/meta/Known%20watermarks.md
WATERMARK_V1_IDS = {"1000000001", "1000000010"}
# Real Stack Overflow post ids are far below this; anything above is fabricated.
MAX_REAL_POST_ID = 1_000_000_000

LIBRARY_TAGS = {"pandas": "pandas", "numpy": "numpy", "matplotlib": "matplotlib", "scikit-learn": "scikit-learn",
                "sklearn": "scikit-learn", "scipy": "scipy"}
MAX_ANSWER_CHARS = 4000

app = typer.Typer(help="Build Stack Overflow chunks")


def is_watermark(row: Mapping[str, str]) -> bool:
    """True for a Posts.xml row (its XML attributes) that matches a known watermark pattern."""
    post_id = row.get("Id", "")
    if post_id in WATERMARK_V1_IDS:
        return True
    return post_id.isdigit() and int(post_id) >= MAX_REAL_POST_ID and row.get("OwnerUserId") == "-1"


def parse_tags(raw: str) -> list[str]:
    """Both dump formats: '<python><pandas>' and '|python|pandas|'."""
    return [t for t in raw.replace("<", "|").replace(">", "|").split("|") if t]


def library_of(tags: list[str]) -> str | None:
    if "python" not in tags and not any(t.startswith("python-") for t in tags):
        return None
    return next((LIBRARY_TAGS[t] for t in tags if t in LIBRARY_TAGS), None)


def license_for(row: Mapping[str, str], created: date) -> str:
    """ContentLicense when the dump has it, else the license Stack Overflow used at that date."""
    if row.get("ContentLicense"):
        return row["ContentLicense"]
    if created < date(2011, 4, 8):
        return "CC BY-SA 2.5"
    return "CC BY-SA 3.0" if created < date(2018, 5, 2) else "CC BY-SA 4.0"


def html_to_text(html: str) -> str:
    """Paragraph text; <pre> blocks become ```python fences and inline <code> becomes `code`."""
    if not html.strip():
        return ""
    root = lxml.html.fragment_fromstring(html, create_parent="div")
    for pre in list(root.iter("pre")):
        code = pre.text_content().rstrip("\n")
        tail = pre.tail
        pre.clear()
        pre.text, pre.tail = f"\n```python\n{code}\n```\n", tail
    for code in list(root.iter("code")):
        parent = code.getparent()
        if parent is not None and parent.tag != "pre":
            text, tail = code.text_content(), code.tail
            code.clear()
            code.text, code.tail = f"`{text}`", tail
    for el in root.iter("p", "li", "h1", "h2", "h3", "blockquote", "br"):
        el.tail = "\n" + (el.tail or "")
    lines = [line.rstrip() for line in root.text_content().splitlines()]
    return "\n".join(lines).strip().replace("\n\n\n", "\n\n")


def _rows(path: Path) -> Iterator[dict[str, str]]:
    for _, el in etree.iterparse(str(path), events=("end",), tag="row"):
        yield dict(el.attrib)
        el.clear()
        while el.getprevious() is not None:
            del el.getparent()[0]


def iter_pairs(path: Path, min_score: int = 3) -> Iterator[tuple[dict, dict, str]]:
    """(question, accepted answer, library). Answers come after their question in the dump (ordered by Id)."""
    pending: dict[str, tuple[dict, str]] = {}  # accepted answer id -> (question, library)
    for row in _rows(path):
        if is_watermark(row):
            continue
        if row.get("PostTypeId") == "1":
            library = library_of(parse_tags(row.get("Tags", "")))
            if library and row.get("AcceptedAnswerId") and int(row.get("Score", 0)) >= min_score:
                pending[row["AcceptedAnswerId"]] = (row, library)
        elif row.get("PostTypeId") == "2" and row.get("Id") in pending:
            question, library = pending.pop(row["Id"])
            yield question, row, library


def make_chunk(question: dict, answer: dict, library: str, infer=None) -> Chunk:
    created = datetime.fromisoformat(answer["CreationDate"]).date()
    q_text, a_text = html_to_text(question.get("Body", "")), html_to_text(answer.get("Body", ""))
    text = f"Q: {question.get('Title', '')}\n\n{q_text}\n\nA:\n{a_text[:MAX_ANSWER_CHARS]}"
    version, inferred = infer(library, text, created) if infer else (None, False)
    url = f"https://stackoverflow.com/a/{answer['Id']}"
    owner = answer.get("OwnerDisplayName") or (f"user {answer['OwnerUserId']}" if answer.get("OwnerUserId") else None)
    return Chunk(
        id=make_chunk_id(url), text=text, source="stackoverflow", library=library, version=version,
        version_inferred=inferred, url=url, title=question.get("Title", ""), score=int(answer.get("Score", 0)),
        created=created, license=license_for(answer, created), author=owner,
    )


@app.command()
def build(
    posts_xml: Path, limit: int | None = None, min_score: int = 3, out: Path = CHUNKS_DIR / "so_all.parquet"
) -> None:
    """Write so_all.parquet; decontam turns it into so_clean.parquet and so_leaky.parquet."""
    from tqdm import tqdm

    from snakepp.corpus.versions import infer_version

    chunks = []
    for question, answer, library in tqdm(iter_pairs(posts_xml, min_score), unit="pair", total=limit):
        chunks.append(make_chunk(question, answer, library, infer_version))
        if limit and len(chunks) >= limit:
            break
    typer.echo(f"{out}: {write_chunks(chunks, out)} chunks")


if __name__ == "__main__":
    app()
