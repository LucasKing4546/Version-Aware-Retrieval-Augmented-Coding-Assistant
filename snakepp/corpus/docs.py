"""Docs corpus — owner: Lucas.

For one library at one version, builds data/chunks/docs_<library>_<version>.parquet from
1. API docstrings, dumped by apidoc_dump.py inside data/envs/docs/<library>-<version> (uv, Python 3.10), and
2. the narrative Sphinx .rst pages (user guide, release notes) from a sparse git checkout of the docs folder.
Usage: python -m snakepp.corpus.docs build pandas 2.2.2      |      python -m snakepp.corpus.docs build-all
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import typer

from snakepp.corpus import rst
from snakepp.corpus.io import CHUNKS_DIR, write_chunks
from snakepp.schemas import Chunk, make_chunk_id

RAW_DIR = Path("data/raw/docs")
ENVS_DIR = Path("data/envs/docs")
APIDOC = Path(__file__).with_name("apidoc_dump.py")
MIN_CHUNK_CHARS = 40
RELEASE_NOTES = re.compile(r"(whatsnew|whats_new|release|api_changes|prev_whats_new)")
_PAGE_VERSION = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?:\.\d+)?(?![\d])")


@dataclass(frozen=True)
class LibrarySpec:
    repo: str
    doc_dir: str
    tag: str  # format string with {v}
    rst_url: str  # format string with {v}, {mm} (major.minor), {page}
    api_url: str  # format string with {v}, {mm}, {name}
    license: str
    modules: tuple[str, ...] = ()
    rst_skip: tuple[str, ...] = ("development/", "getting_started/install", "conf.py")
    extra_requirements: dict[str, str] = field(default_factory=dict)  # version -> extra pip spec


SPECS: dict[str, LibrarySpec] = {
    "pandas": LibrarySpec(
        "https://github.com/pandas-dev/pandas", "doc/source", "v{v}",
        "https://pandas.pydata.org/pandas-docs/version/{v}/{page}.html",
        "https://pandas.pydata.org/pandas-docs/version/{v}/reference/api/{name}.html", "BSD-3-Clause",
        ("pandas", "pandas.api.types", "pandas.core.groupby", "pandas.core.window", "pandas.core.resample"),
        extra_requirements={"1.5.3": "numpy<2"},
    ),
    "numpy": LibrarySpec(
        "https://github.com/numpy/numpy", "doc/source", "v{v}", "https://numpy.org/doc/{mm}/{page}.html",
        "https://numpy.org/doc/{mm}/reference/generated/{name}.html", "BSD-3-Clause",
        ("numpy", "numpy.linalg", "numpy.random", "numpy.fft", "numpy.ma", "numpy.polynomial", "numpy.char"),
    ),
    "matplotlib": LibrarySpec(
        "https://github.com/matplotlib/matplotlib", "doc", "v{v}", "https://matplotlib.org/{v}/{page}.html",
        "https://matplotlib.org/{v}/api/_as_gen/{name}.html", "PSF-based Matplotlib License",
        ("matplotlib", "matplotlib.pyplot", "matplotlib.axes", "matplotlib.figure", "matplotlib.colors",
         "matplotlib.cm", "matplotlib.ticker", "matplotlib.dates", "matplotlib.patches", "matplotlib.lines"),
        extra_requirements={"3.8.4": "numpy<2"},
    ),
    "scikit-learn": LibrarySpec(
        "https://github.com/scikit-learn/scikit-learn", "doc", "{v}", "https://scikit-learn.org/{mm}/{page}.html",
        "https://scikit-learn.org/{mm}/modules/generated/{name}.html", "BSD-3-Clause",
        tuple(f"sklearn.{m}" for m in (
            "preprocessing", "model_selection", "metrics", "linear_model", "ensemble", "cluster", "decomposition",
            "pipeline", "compose", "impute", "neighbors", "svm", "tree", "feature_extraction.text", "datasets",
            "feature_selection", "naive_bayes", "manifold", "utils")),
        extra_requirements={"1.4.0": "numpy<2"},
    ),
    "scipy": LibrarySpec(
        "https://github.com/scipy/scipy", "doc/source", "v{v}",
        "https://docs.scipy.org/doc/scipy-{v}/{page}.html",
        "https://docs.scipy.org/doc/scipy-{v}/reference/generated/{name}.html", "BSD-3-Clause",
        tuple(f"scipy.{m}" for m in (
            "stats", "optimize", "integrate", "interpolate", "linalg", "sparse", "signal", "spatial", "special",
            "ndimage", "fft", "cluster.hierarchy", "cluster.vq", "io")),
        extra_requirements={"1.12.0": "numpy<2"},
    ),
    "python": LibrarySpec(
        "https://github.com/python/cpython", "Doc", "v{v}", "https://docs.python.org/{mm}/{page}.html", "",
        "PSF-2.0", rst_skip=("c-api/", "extending/", "distributing/", "installing/", "using/", "faq/", "howto/",
                              "includes/", "tools/", "conf.py"),
    ),
}
DEFAULT_VERSIONS = {
    "pandas": ["1.5.3", "2.2.2"], "numpy": ["1.26.4", "2.0.2"], "matplotlib": ["3.8.4", "3.9.2"],
    "scikit-learn": ["1.4.0", "1.5.2"], "scipy": ["1.12.0", "1.13.1"], "python": ["3.10.14"],
}

app = typer.Typer(help="Build documentation chunks")


def _mm(version: str) -> str:
    return ".".join(version.split(".")[:2])


def label(library: str, version: str) -> str:
    """Version stored on chunks: full version, except major.minor for the Python stdlib."""
    return _mm(version) if library == "python" else version


def checkout_docs(spec: LibrarySpec, version: str, root: Path = RAW_DIR) -> Path:
    dest = root / f"{spec.repo.rsplit('/', 1)[-1]}-{version}"
    if not (dest / spec.doc_dir).exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", "--branch",
                        spec.tag.format(v=version), spec.repo, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", spec.doc_dir], check=True)
    return dest / spec.doc_dir


def keep_page(page: str, version: str, minor_window: int = 2) -> bool:
    """Release-note pages only for the same major and at most `minor_window` minors back (none from the future)."""
    if not RELEASE_NOTES.search(page):
        return True
    m = _PAGE_VERSION.search(page.replace("_", ".").replace("-", "."))
    if not m:
        return True
    major, minor = (int(x) for x in version.split(".")[:2])
    return int(m.group(1)) == major and minor - minor_window <= int(m.group(2)) <= minor


def rst_chunks(library: str, version: str, doc_root: Path, spec: LibrarySpec) -> Iterator[Chunk]:
    for path in sorted(doc_root.rglob("*.rst")):
        page = path.relative_to(doc_root).with_suffix("").as_posix()
        if any(page.startswith(s) or s in page for s in spec.rst_skip) or not keep_page(page, version):
            continue
        text = rst.clean_rst(path.read_text(encoding="utf-8", errors="replace"))
        url = spec.rst_url.format(v=version, mm=_mm(version), page=page)
        for n, sec in enumerate(rst.pack(rst.split_sections(text))):
            if len(sec.text) < MIN_CHUNK_CHARS:
                continue
            title = " > ".join(sec.path) or page
            yield Chunk(id=make_chunk_id(url, title, str(n)), text=sec.text, source="docs", library=library,
                        version=label(library, version), url=url, title=f"{title} ({page})", license=spec.license)


def ensure_env(library: str, version: str, spec: LibrarySpec, root: Path = ENVS_DIR) -> Path:
    env = root / f"{library}-{version}"
    exe = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not exe.exists():
        uv = [sys.executable, "-m", "uv"]
        subprocess.run([*uv, "venv", "-q", "--python", "3.10", str(env)], check=True)
        reqs = [f"{library}=={version}", *filter(None, [spec.extra_requirements.get(version)])]
        subprocess.run([*uv, "pip", "install", "-q", "--python", str(env), *reqs], check=True)
    return exe


def api_records(library: str, version: str, spec: LibrarySpec) -> list[dict]:
    if not spec.modules:
        return []
    out = RAW_DIR / f"apidoc_{library}_{version}.jsonl"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        root = spec.modules[0].split(".")[0]
        exe = ensure_env(library, version, spec)
        subprocess.run([str(exe), "-I", str(APIDOC.resolve()), str(out.resolve()), root, *spec.modules], check=True)
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]


def api_chunks(library: str, version: str, records: list[dict], spec: LibrarySpec) -> Iterator[Chunk]:
    for rec in records:
        name = rec["qualname"]
        url = spec.api_url.format(v=version, mm=_mm(version), name=name)
        body = f"{name}{rec['signature']}\n\n{rec['doc']}"
        for n, sec in enumerate(rst.pack([rst.Section((name,), body)], max_chars=3000, min_chars=0)):
            yield Chunk(id=make_chunk_id(url, str(n)), text=sec.text, source="docs", library=library,
                        version=label(library, version), url=url, title=name if n == 0 else f"{name} (part {n + 1})",
                        license=spec.license)


@app.command()
def build(library: str, version: str, out_dir: Path = CHUNKS_DIR) -> None:
    """Build docs_<library>_<version>.parquet."""
    spec = SPECS[library]
    chunks = list(api_chunks(library, version, api_records(library, version, spec), spec))
    n_api = len(chunks)
    chunks += rst_chunks(library, version, checkout_docs(spec, version), spec)
    path = out_dir / f"docs_{library}_{label(library, version)}.parquet"
    write_chunks(chunks, path)
    typer.echo(f"{path}: {n_api} API chunks + {len(chunks) - n_api} page chunks")


@app.command("build-all")
def build_all(libraries: list[str] | None = None) -> None:
    """Build every library at its default versions (DS-1000 pins + one newer release)."""
    for library in libraries or DEFAULT_VERSIONS:
        for version in DEFAULT_VERSIONS[library]:
            build(library, version)


if __name__ == "__main__":
    app()
