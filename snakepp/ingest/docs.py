"""Fetch the built HTML docs of one library at one version.

The page list comes from the Sphinx `objects.inv` the docs site publishes. We use built HTML rather than the
.rst source because API pages are generated from docstrings at build time, and because gold sets key chunks by
the real site URL + heading path. Raw pages are cached under data/raw/docs/<library>/<version>/, so reruns
only fetch what is missing.
"""

from __future__ import annotations

import re
import time
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError

import lxml.html

from snakepp.ingest.pypi import USER_AGENT
from snakepp.paths import DATA

RAW_DIR = DATA / "raw" / "docs"
DOCS_BASE = {
    "pandas": "https://pandas.pydata.org/pandas-docs/version/{version}/",
}
DOCS_LICENSE = {
    "pandas": "BSD-3-Clause",
}
# Sphinx inventory v2 line: name domain:role priority uri dispname
INV_LINE = re.compile(r"(.+?)\s+(\S+)\s+(-?\d+)\s+?(\S*)\s+(.*)")
SKIP_PAGES = {"genindex.html", "py-modindex.html", "search.html"}  # generated indexes, no prose

HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Elements that end a line of text; without this, text_content() glues "Parameters" + "other" + ...
LINE_TAGS = HEADINGS | {"p", "div", "dl", "dt", "dd", "ul", "ol", "li", "table", "tr", "blockquote", "pre", "section"}


@dataclass
class Section:
    """One docs section: its heading path from the page title down, its anchor, and its blocks in order."""

    heading_path: tuple[str, ...]
    anchor: str
    blocks: list[tuple[str, str]] = field(default_factory=list)  # ("text" | "code", content)


def _clean(text: str) -> str:
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _blocks(el, out: list[tuple[str, str]]) -> None:
    """Append el's content as blocks: each <pre> whole as code, text around it as text."""
    if el.tag == "pre":
        if code := el.text_content().strip("\n").rstrip():
            out.append(("code", code))
    elif el.find(".//pre") is not None:  # mixed container (API dl, admonition with code): recurse
        if el.text and el.text.strip():
            out.append(("text", _clean(el.text)))
        for child in el:
            if isinstance(child.tag, str):
                _blocks(child, out)
            if child.tail and child.tail.strip():
                out.append(("text", _clean(child.tail)))
    elif text := _clean(el.text_content()):
        out.append(("text", text))


def _walk(sec, path: tuple[str, ...], out: list[Section]) -> None:
    section = Section(path, sec.get("id") or "")
    subsections, titled = [], False
    for child in sec:
        if not isinstance(child.tag, str):  # comments, processing instructions
            continue
        if child.tag == "section":
            subsections.append(child)
        elif child.tag in HEADINGS and not titled:
            section.heading_path, titled = path + (_clean(child.text_content()),), True
        else:
            _blocks(child, section.blocks)
    out.append(section)
    for sub in subsections:
        _walk(sub, section.heading_path, out)


def _prepare(root) -> None:
    """Drop link/script noise and add line breaks after block elements, so text_content() reads naturally."""
    noise = ".//a[contains(@class, 'headerlink')] | .//span[contains(@class, 'viewcode-link')] | .//script | .//style"
    for el in root.xpath(noise):
        el.drop_tree()
    for el in root.xpath(".//span[contains(@class, 'classifier')]"):  # "other : DataFrame", not "otherDataFrame"
        el.text = " : " + (el.text or "")
    for el in root.iter(*LINE_TAGS):
        el.tail = "\n" + (el.tail or "")
    for el in root.iter("td", "th"):
        el.tail = " " + (el.tail or "")


def html_blocks(html: str) -> list[tuple[str, str]]:
    """Blocks of an HTML fragment (e.g. a Stack Overflow post body), same rules as docs sections."""
    root = lxml.html.fragment_fromstring(html, create_parent="div")
    _prepare(root)
    out: list[tuple[str, str]] = []
    _blocks(root, out)
    return out


def parse_page(html: bytes | str) -> list[Section]:
    """Split a built Sphinx (pydata theme) page into heading-scoped sections, in document order."""
    root = lxml.html.fromstring(html)
    article = (
        root.xpath("//article[contains(concat(' ', @class, ' '), ' bd-article ')]")
        or root.xpath("//*[@role='main']")
        or [root]
    )[0]
    _prepare(article)
    out: list[Section] = []
    _walk(article, (), out)
    return [s for s in out if s.blocks]


def inventory_pages(blob: bytes) -> list[str]:
    """Unique page paths (anchor dropped) listed in a Sphinx v2 objects.inv."""
    *header, body = blob.split(b"\n", 4)
    if header[0].strip() != b"# Sphinx inventory version 2":
        raise ValueError(f"not a Sphinx v2 inventory: {header[0][:60]!r}")
    pages = set()
    for line in zlib.decompress(body).decode("utf-8").splitlines():
        if m := INV_LINE.match(line):
            name, uri = m.group(1), m.group(4)
            if uri.endswith("$"):  # inventory shorthand for "uri + name"
                uri = uri[:-1] + name
            pages.add(uri.split("#")[0])
    return sorted(p for p in pages if p.endswith(".html") and p not in SKIP_PAGES)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_pages(library: str, version: str, delay_s: float = 0.2, out_dir: Path = RAW_DIR) -> Path:
    """Download every page of the docs inventory that is not cached yet. Returns the version's raw dir."""
    base = DOCS_BASE[library].format(version=version)
    root = out_dir / library / version
    root.mkdir(parents=True, exist_ok=True)
    inv = root / "objects.inv"
    if not inv.exists():
        inv.write_bytes(_get(base + "objects.inv"))
    pages = inventory_pages(inv.read_bytes())
    fetched, missing = 0, []
    for page in pages:
        path = root / page
        if path.exists():
            continue
        try:
            body = _get(base + page)
        except HTTPError as e:
            if e.code != 404:
                raise
            missing.append(page)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        fetched += 1
        if fetched % 200 == 0:
            print(f"{library} {version}: {fetched} fetched")
        time.sleep(delay_s)  # be polite to the docs host
    print(f"{library} {version}: {len(pages)} pages, {fetched} new, {len(missing)} missing (404)")
    return root
