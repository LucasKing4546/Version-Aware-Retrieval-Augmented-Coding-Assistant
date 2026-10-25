import json
import subprocess
import sys

from snakepp.corpus import docs

SPEC = docs.SPECS["pandas"]


def test_keep_page_release_window():
    assert docs.keep_page("user_guide/merging", "2.2.2")
    assert docs.keep_page("whatsnew/v2.0.0", "2.2.2")
    assert not docs.keep_page("whatsnew/v0.13.1", "2.2.2")
    assert not docs.keep_page("whatsnew/v2.3.0", "2.2.2")  # from the future
    assert docs.keep_page("release/1.25.0-notes", "1.26.4") and not docs.keep_page("release/1.20.0-notes", "1.26.4")


def test_label():
    assert docs.label("python", "3.10.14") == "3.10" and docs.label("pandas", "2.2.2") == "2.2.2"


def test_rst_chunks(tmp_path):
    (tmp_path / "whatsnew").mkdir()
    (tmp_path / "whatsnew" / "v2.0.0.rst").write_text(
        "What's new\n==========\n\nRemoved :meth:`DataFrame.append`; use :func:`concat`.\n" + "More text. " * 10,
        encoding="utf-8",
    )
    (tmp_path / "whatsnew" / "v0.13.1.rst").write_text("Old\n===\n\nAncient release notes text here.\n")
    chunks = list(docs.rst_chunks("pandas", "2.2.2", tmp_path, SPEC))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.version == "2.2.2" and c.source == "docs" and "DataFrame.append" in c.text
    assert c.url == "https://pandas.pydata.org/pandas-docs/version/2.2.2/whatsnew/v2.0.0.html"


def test_api_chunks_split_long_docstrings():
    records = [{"qualname": "pandas.concat", "signature": "(objs, axis=0)", "doc": "Concatenate.\n\n" + "x " * 2000}]
    chunks = list(docs.api_chunks("pandas", "2.2.2", records, SPEC))
    assert len(chunks) == 2 and chunks[0].title == "pandas.concat" and chunks[1].title == "pandas.concat (part 2)"
    assert chunks[0].text.startswith("pandas.concat(objs, axis=0)")
    assert chunks[0].url.endswith("/2.2.2/reference/api/pandas.concat.html")
    assert len({c.id for c in chunks}) == 2


def test_apidoc_dump_runs_standalone(tmp_path):
    out = tmp_path / "api.jsonl"
    subprocess.run([sys.executable, "-I", str(docs.APIDOC), str(out), "json", "json"], check=True)
    names = {json.loads(line)["qualname"] for line in out.read_text(encoding="utf-8").splitlines()}
    assert {"json.dumps", "json.JSONDecoder", "json.JSONDecoder.decode"} <= names


def test_apidoc_dump_skips_missing_modules(tmp_path):
    out = tmp_path / "api.jsonl"
    subprocess.run([sys.executable, "-I", str(docs.APIDOC), str(out), "json", "json.nope", "json"], check=True)
    assert out.exists() and not (tmp_path / "api.jsonl.part").exists()
