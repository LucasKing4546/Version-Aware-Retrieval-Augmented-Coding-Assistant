import json
import zlib
from datetime import date
from pathlib import Path

import pytest

from snakepp.ingest import so_fetch
from snakepp.ingest.docs import inventory_pages, parse_page
from snakepp.ingest.pypi import release_dates, version_at


def _file(day: str, yanked: bool = False) -> dict:
    return {"upload_time": f"{day}T10:00:00", "yanked": yanked}


def test_release_dates_keeps_final_unyanked_releases_sorted():
    raw = {
        "2.0.0": [_file("2023-04-03")],
        "2.0.0rc0": [_file("2023-03-01")],
        "1.5.3": [_file("2023-01-19"), _file("2023-01-18")],
        "1.5.4": [_file("2023-02-01", yanked=True)],
        "1.10.0": [_file("2022-01-01")],
        "0.1": [],
        "not-a-version": [_file("2020-01-01")],
    }
    assert release_dates(raw) == {"1.5.3": "2023-01-18", "1.10.0": "2022-01-01", "2.0.0": "2023-04-03"}


def test_version_at_picks_highest_released_version():
    releases = {"1.5.2": "2022-11-22", "1.5.3": "2023-06-01", "2.0.0": "2023-04-03"}
    assert version_at(releases, date(2022, 1, 1)) is None
    assert version_at(releases, date(2023, 1, 1)) == "1.5.2"
    assert version_at(releases, date(2023, 7, 1)) == "2.0.0"  # a late 1.x backport doesn't win


def test_inventory_pages():
    body = "\n".join([
        "pandas.DataFrame.append py:method 1 reference/api/pandas.DataFrame.append.html#$ -",
        "pandas.DataFrame.append py:method 1 reference/api/pandas.DataFrame.append.html#x -",
        "user_guide/merging std:doc -1 user_guide/merging.html Merge, join",
        "merging-concat std:label -1 user_guide/merging.html#merging-concat Concatenating",
        "genindex std:label -1 genindex.html Index",
        "weird std:label -1 #only-anchor -",
    ])
    blob = b"# Sphinx inventory version 2\n# Project: x\n# Version: 1\n# zlib\n" + zlib.compress(body.encode())
    assert inventory_pages(blob) == [
        "reference/api/pandas.DataFrame.append.html",
        "user_guide/merging.html",
    ]
    with pytest.raises(ValueError):
        inventory_pages(b"# Sphinx inventory version 1\na\nb\nc\nd")


def test_fetch_questions_walks_dates_filters_and_resumes(tmp_path, monkeypatch):
    q1 = {"question_id": 1, "score": 5, "accepted_answer_id": 11, "creation_date": 100}
    q2 = {"question_id": 2, "score": 1, "accepted_answer_id": 12, "creation_date": 200}  # score too low
    q3 = {"question_id": 3, "score": 9, "accepted_answer_id": 13, "creation_date": 300}
    pages = {0: [q1, q2], 200: [q2, q3]}  # inclusive fromdate re-returns q2
    calls = []

    def fake_get(path, **params):
        calls.append(path)
        if path.startswith("/answers/"):
            ids = path.removeprefix("/answers/").split(";")
            return {"items": [{"answer_id": int(i)} for i in ids], "quota_remaining": 50}
        items = pages[params["fromdate"]]
        return {"items": items, "has_more": params["fromdate"] == 0, "quota_remaining": 50}

    monkeypatch.setattr(so_fetch, "_get", fake_get)
    assert so_fetch.fetch_questions("pandas", date(1970, 1, 1), out_dir=tmp_path) == 2

    first = json.loads((tmp_path / "pandas" / "0.json").read_text())
    assert [q["question_id"] for q in first["questions"]] == [1]
    assert [a["answer_id"] for a in first["answers"]] == [11]
    assert json.loads((tmp_path / "pandas" / "state.json").read_text()) == {"fromdate": 300, "done": True}

    calls.clear()
    assert so_fetch.fetch_questions("pandas", date(1970, 1, 1), out_dir=tmp_path) == 0
    assert calls == []


PAGE = """<html><body><nav>sidebar noise</nav>
<article class="bd-article"><section id="pandas-dataframe-append">
<h1>pandas.DataFrame.append<a class="headerlink" href="#">#</a></h1>
<dl class="py method"><dt class="sig">DataFrame.append(other, ignore_index=False)<a class="reference external">\
<span class="viewcode-link"><span class="pre">[source]</span></span></a></dt><dd>
<p>Append rows of <code>other</code> to the end of caller.</p>
<div class="deprecated"><p>Deprecated since version 1.4.0: Use <code>concat()</code> instead.</p></div>
<dl class="field-list"><dt>Parameters</dt><dd><dl>
<dt><strong>other</strong><span class="classifier">DataFrame or Series</span></dt><dd><p>The data.</p></dd>
</dl></dd></dl>
<p class="rubric">Examples</p>
<div class="doctest"><div class="highlight"><pre>&gt;&gt;&gt; df.append(df2)
   A  B

0  1  2</pre></div></div>
<div class="highlight"><pre>  </pre></div>
<p>After the example.</p>
</dd></dl>
<section id="concat"><h2>concat()<a class="headerlink">#</a></h2><p>Use concat.</p>
<section id="empty"><h3>Only a heading</h3></section>
</section></section></article></body></html>"""


def test_parse_page_sections_blocks_and_code():
    sections = parse_page(PAGE)
    assert [(s.heading_path, s.anchor) for s in sections] == [
        (("pandas.DataFrame.append",), "pandas-dataframe-append"),
        (("pandas.DataFrame.append", "concat()"), "concat"),
    ]
    blocks = sections[0].blocks
    assert ("code", ">>> df.append(df2)\n   A  B\n\n0  1  2") in blocks  # whitespace kept, empty <pre> dropped
    text = "\n".join(t for kind, t in blocks if kind == "text")
    assert "DataFrame.append(other, ignore_index=False)" in text
    assert "Append rows of other to the end of caller." in text
    assert "Deprecated since version 1.4.0: Use concat() instead." in text
    assert "other : DataFrame or Series" in text and "Parameters\n" in text
    assert "sidebar" not in text and "#" not in text and "[source]" not in text
    assert [k for k, _ in blocks][-2:] == ["code", "text"]  # document order kept around the code
    assert sections[1].blocks == [("text", "Use concat.")]


def test_chunk_sections_caps_size_keeps_code_whole_and_ids_stable():
    from snakepp.ingest.chunker import MAX_CHARS, chunk_sections
    from snakepp.ingest.docs import Section

    big_code = "\n".join(f"df{i} = pd.concat([df, row{i}])" for i in range(120))  # > MAX_CHARS on its own
    long_text = "\n".join(f"Parameter {i} : int, default {i}. Explains option {i}." for i in range(80))
    sections = [
        Section(("pandas.concat",), "pandas-concat", [("text", "Concatenate objects."), ("text", long_text)]),
        Section(("pandas.concat", "Examples"), "ex", [("code", big_code), ("text", "Done.")]),
        Section(("pandas.concat", "Examples"), "ex2", [("text", "Same heading path, other section.")]),
    ]
    url = "https://pandas.pydata.org/pandas-docs/version/2.2.3/reference/api/pandas.concat.html"
    kw = dict(url=url, library="pandas", version="2.2.3", license="BSD-3-Clause")
    chunks = chunk_sections(sections, **kw)

    assert chunks == chunk_sections(sections, **kw)  # deterministic ids and text
    assert len({c.id for c in chunks}) == len(chunks)
    assert {c.title for c in chunks} == {"pandas.concat"}
    assert chunks[0].text.startswith("pandas > 2.2.3 > pandas.concat\n\nConcatenate objects.")
    assert len(chunks[0].text) <= MAX_CHARS and chunks[1].heading_path == "pandas.concat"  # long text was split
    code_chunks = [c for c in chunks if "df0 = " in c.text]
    assert len(code_chunks) == 1 and f"```\n{big_code}\n```" in code_chunks[0].text  # code never split
    assert all(len(c.text) <= MAX_CHARS for c in chunks if c is not code_chunks[0])
    assert [c.heading_path for c in chunks].count("pandas.concat > Examples") == 3
    assert all(c.source == "docs" and c.version == "2.2.3" and not c.version_inferred for c in chunks)


def test_load_and_write_chunks_roundtrip(tmp_path, chunks_200):
    from snakepp.ingest.chunks import SCHEMA, load_chunks, write_chunks
    from snakepp.schemas import Chunk

    assert SCHEMA.names == list(Chunk.model_fields)  # parquet schema follows the contract
    docs_only = [c.model_copy(update={"score": None, "created": None, "author": None}) for c in chunks_200[:3]]
    p1 = write_chunks(chunks_200, tmp_path / "a" / "all.parquet")
    p2 = write_chunks(docs_only, tmp_path / "docs.parquet")  # all-null columns keep their types
    assert list(load_chunks([p1])) == chunks_200
    assert list(load_chunks(str(p2))) == docs_only  # a lone path works too

    jsonl = tmp_path / "f.jsonl"
    jsonl.write_text("\n".join(c.model_dump_json() for c in chunks_200[:5]) + "\n\n", encoding="utf-8")
    assert list(load_chunks([jsonl, p2])) == chunks_200[:5] + docs_only


def test_so_qa_chunk_and_dedupe(tmp_path):
    from snakepp.ingest.so_parse import iter_qa, qa_chunk

    q = {"question_id": 7, "accepted_answer_id": 70, "title": "Can&#39;t append", "body": "<p>How?</p>"}
    a = {
        "answer_id": 70, "creation_date": 1675209600, "score": 12, "content_license": "CC BY-SA 4.0",  # 2023-02-01
        "owner": {"display_name": "J&amp;K"},
        "body": "<p>Use <code>concat</code>:</p><pre><code>pd.concat([df, row])\n</code></pre>",
    }  # fmt: skip
    (tmp_path / "100.json").write_text(json.dumps({"questions": [q], "answers": [a]}))
    (tmp_path / "200.json").write_text(json.dumps({"questions": [q], "answers": [a]}))  # inclusive-cursor repeat
    (tmp_path / "state.json").write_text("{}")
    pairs = list(iter_qa(tmp_path))
    assert len(pairs) == 1

    c = qa_chunk(*pairs[0], "pandas", {"1.5.3": "2023-01-19", "2.0.0": "2023-04-03"})
    assert (c.version, c.version_inferred, c.created) == ("1.5.3", True, date(2023, 2, 1))
    assert (c.title, c.author, c.score, c.license) == ("Can't append", "J&K", 12, "CC BY-SA 4.0")
    assert c.url == "https://stackoverflow.com/questions/7" and c.source == "stackoverflow"
    assert c.text == (
        "Can't append\n\nQuestion:\nHow?\n\nAccepted answer:\nUse concat:\n\n```\npd.concat([df, row])\n```"
    )


def test_real_fixture_has_the_version_trap():
    from snakepp.ingest.chunks import load_chunks

    chunks = list(load_chunks(Path(__file__).parent / "fixtures" / "fixture_chunks.jsonl"))
    assert 150 <= len(chunks) <= 250 and len({c.id for c in chunks}) == len(chunks)
    docs = [c for c in chunks if c.source == "docs"]
    assert {c.version for c in docs} == {"1.5.3", "2.2.3"} and all(c.heading_path for c in docs)
    append = {c.version for c in docs if c.url.endswith("/pandas.DataFrame.append.html")}
    assert append == {"1.5.3"}  # removed in 2.0: no page in the 2.2.3 docs
    assert any(c.url.endswith("/pandas.DataFrame.map.html") for c in docs if c.version == "2.2.3")
    so = [c for c in chunks if c.source == "stackoverflow"]
    assert len(so) >= 10 and all(c.version_inferred and c.author and c.created for c in so)
    assert any(".append(" in c.text for c in so)
