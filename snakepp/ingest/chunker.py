"""Heading-aware docs chunker: each parsed Section becomes one or more Chunks.

Sections are never merged, so a gold ref (url + heading_path) maps to the chunks of exactly one section. A section
over the size cap is split between blocks; a text block over the cap is split at line breaks; code is never split.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from snakepp.ingest.docs import DOCS_BASE, DOCS_LICENSE, RAW_DIR, Section, inventory_pages, parse_page
from snakepp.schemas import HEADING_SEP, Chunk, make_chunk_id

# ponytail: ~512 tokens at ~4 chars/token; switch to the embedder's tokenizer if chunks overflow at index time.
MAX_CHARS = 2000


def _pieces(blocks: list[tuple[str, str]], budget: int) -> Iterator[str]:
    """Rendered blocks; text longer than `budget` is split at line breaks, code is fenced and kept whole."""
    for kind, text in blocks:
        if kind == "code":
            yield f"```\n{text}\n```"
            continue
        lines: list[str] = []
        for line in text.splitlines():
            if lines and len("\n".join([*lines, line])) > budget:
                yield "\n".join(lines)
                lines = []
            lines.append(line)
        if lines:
            yield "\n".join(lines)


def chunk_sections(sections: list[Section], *, url: str, library: str, version: str, license: str) -> list[Chunk]:
    """Chunks for one page, in document order. Ids hash url + heading path + position on the page."""
    title = next((s.heading_path[0] for s in sections if s.heading_path), url.rsplit("/", 1)[-1])
    chunks: list[Chunk] = []
    for sec in sections:
        heading_path = HEADING_SEP.join(sec.heading_path)
        header = HEADING_SEP.join([library, version, *sec.heading_path])
        budget = MAX_CHARS - len(header) - 2
        groups: list[list[str]] = []
        for piece in _pieces(sec.blocks, budget):
            if groups and len("\n\n".join([*groups[-1], piece])) <= budget:
                groups[-1].append(piece)
            else:
                groups.append([piece])
        for group in groups:
            chunks.append(
                Chunk(
                    id=make_chunk_id(url, heading_path, str(len(chunks))),
                    text=header + "\n\n" + "\n\n".join(group),
                    source="docs",
                    library=library,
                    version=version,
                    url=url,
                    title=title,
                    heading_path=heading_path,
                    license=license,
                )
            )
    return chunks


def chunk_docs(library: str, version: str, raw_dir: Path = RAW_DIR) -> list[Chunk]:
    """Chunk every cached page of one docs version (pages not downloaded yet are skipped and counted)."""
    root = raw_dir / library / version
    base = DOCS_BASE[library].format(version=version)
    chunks, missing = [], 0
    for page in inventory_pages((root / "objects.inv").read_bytes()):
        path = root / page
        if not path.exists():
            missing += 1
            continue
        sections = parse_page(path.read_bytes())
        chunks += chunk_sections(sections, url=base + page, library=library, version=version,
                                 license=DOCS_LICENSE[library])  # fmt: skip
    if missing:
        print(f"{library} {version}: {missing} pages not downloaded yet, skipped")
    return chunks
