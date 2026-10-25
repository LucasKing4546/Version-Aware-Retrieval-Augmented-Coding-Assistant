"""Split reStructuredText (Sphinx) files into retrieval-sized sections.

Sections follow headings; tiny sections are merged into the previous one and long ones are split at blank lines
outside indented (code / directive) blocks, so a code block is never cut in half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+<>_:.])\1{2,}\s*$")
_ROLE = re.compile(r":[\w:.+-]+:`~?([^`<]*?)(?:\s*<([^>]+)>)?`")
_TARGET = re.compile(r"^\.\. _[^:]+:\s*$", re.MULTILINE)
# ".. text" comments and their indented body; directives (".. name::") never match
_COMMENT = re.compile(r"^\.\.(?: [^\n:]*)?\n(?:[ \t]+[^\n]*\n)*", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    path: tuple[str, ...]  # heading path, outermost first
    text: str


def clean_rst(text: str) -> str:
    """Drop link targets and comments; turn roles like :func:`~pandas.concat` into their target text."""
    text = _TARGET.sub("", text)
    text = _COMMENT.sub("", text)
    text = _ROLE.sub(lambda m: (m.group(1) or m.group(2) or "").strip(), text)
    return re.sub(r"\n{3,}", "\n\n", text)


def split_sections(text: str, title: str = "") -> list[Section]:
    """Sections by heading. Heading levels are assigned in order of first appearance, as docutils does."""
    lines = text.splitlines()
    levels: list[str] = []
    path: list[str] = [title] if title else []
    base = len(path)
    sections: list[Section] = []
    body: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        is_heading = line.strip() and not line.startswith(" ") and _UNDERLINE.match(nxt) and len(nxt.strip()) >= len(
            line.strip()
        )
        if is_heading and not _UNDERLINE.match(line):
            if "".join(body).strip():
                sections.append(Section(tuple(path), "\n".join(body).strip()))
            body = []
            char = nxt.strip()[0]
            if char not in levels:
                levels.append(char)
            depth = levels.index(char)
            path = path[: base + depth] + [line.strip()]
            i += 2
            continue
        if _UNDERLINE.match(line) and nxt.strip() and i + 2 < len(lines) and _UNDERLINE.match(lines[i + 2]):
            i += 1  # overline: the title follows on the next line
            continue
        body.append(line)
        i += 1
    if "".join(body).strip():
        sections.append(Section(tuple(path), "\n".join(body).strip()))
    return sections


def _blocks(text: str) -> list[str]:
    """Paragraph blocks; an indented block stays attached to the paragraph that introduces it."""
    blocks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        if blocks and (para.startswith((" ", "\t")) or blocks[-1].rstrip().endswith("::")):
            blocks[-1] += "\n\n" + para
        else:
            blocks.append(para)
    return blocks


def _hard_split(block: str, max_chars: int) -> list[str]:
    """Cut a single oversized block at line boundaries (last resort, may split a code block)."""
    if len(block) <= max_chars:
        return [block]
    pieces, current = [], ""
    for line in block.splitlines(keepends=True):
        if current and len(current) + len(line) > max_chars:
            pieces.append(current)
            current = ""
        current += line[:max_chars]
    return [*pieces, current] if current else pieces


def pack(sections: list[Section], max_chars: int = 3000, min_chars: int = 400) -> list[Section]:
    """Merge sections shorter than min_chars into the previous one; split those longer than max_chars."""
    merged: list[Section] = []
    for s in sections:
        if merged and len(s.text) < min_chars and len(merged[-1].text) + len(s.text) <= max_chars:
            heading = " / ".join(s.path[len(merged[-1].path):]) or (s.path[-1] if s.path else "")
            merged[-1] = Section(merged[-1].path, f"{merged[-1].text}\n\n{heading}\n\n{s.text}".strip())
        else:
            merged.append(s)
    out: list[Section] = []
    for s in merged:
        current = ""
        for block in (piece for b in _blocks(s.text) for piece in _hard_split(b, max_chars)):
            if current and len(current) + len(block) > max_chars:
                out.append(Section(s.path, current.strip()))
                current = ""
            current += block + "\n\n"
        if current.strip():
            out.append(Section(s.path, current.strip()))
    return out
