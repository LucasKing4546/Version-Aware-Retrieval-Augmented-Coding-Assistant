from datetime import date
from xml.sax.saxutils import quoteattr

from snakepp import cache
from snakepp.corpus import stackoverflow as so
from snakepp.corpus import versions


def _row(**attrs):
    return "  <row " + " ".join(f"{k}={quoteattr(str(v))}" for k, v in attrs.items()) + " />\n"


def _posts_xml(tmp_path):
    rows = [
        _row(Id=1, PostTypeId=1, AcceptedAnswerId=3, Score=10, Tags="|python|pandas|", Title="Add row to df",
             Body="<p>How do I add a row with <code>df.append</code>?</p>", CreationDate="2019-03-01T10:00:00.000"),
        _row(Id=2, PostTypeId=1, AcceptedAnswerId=5, Score=1, Tags="<python><numpy>", Title="low score",
             Body="<p>x</p>", CreationDate="2019-03-01T10:00:00.000"),
        _row(Id=3, PostTypeId=2, ParentId=1, Score=25, OwnerUserId=42, ContentLicense="CC BY-SA 4.0",
             Body="<p>Use concat:</p><pre><code>df = pd.concat([df, row])\n</code></pre>",
             CreationDate="2019-03-02T10:00:00.000"),
        _row(Id=4, PostTypeId=1, AcceptedAnswerId=1000000010, Score=5, Tags="|python|pandas|", Title="fake",
             Body="<p>x</p>", CreationDate="2025-06-01T01:00:00.100"),
        _row(Id=1000000010, PostTypeId=2, ParentId=4, Score=1, OwnerUserId=-1, Body="<p>bogus</p>",
             CreationDate="2025-06-01T01:03:15.100"),
        _row(Id=6, PostTypeId=1, AcceptedAnswerId=7, Score=9, Tags="|javascript|", Title="js", Body="",
             CreationDate="2019-03-01T10:00:00.000"),
        _row(Id=7, PostTypeId=2, ParentId=6, Score=9, Body="<p>js</p>", CreationDate="2019-03-01T10:00:00.000"),
    ]
    path = tmp_path / "Posts.xml"
    path.write_text('<?xml version="1.0" encoding="utf-8"?>\n<posts>\n' + "".join(rows) + "</posts>\n",
                    encoding="utf-8")
    return path


def test_is_watermark():
    assert so.is_watermark({"Id": "1000000001", "OwnerUserId": "-1"})
    assert so.is_watermark({"Id": "1000000123", "OwnerUserId": "-1"})
    assert not so.is_watermark({"Id": "1000000123", "OwnerUserId": "42"})
    assert not so.is_watermark({"Id": "11227809", "OwnerUserId": "-1"})


def test_tags_and_library():
    assert so.parse_tags("<python><pandas>") == so.parse_tags("|python|pandas|") == ["python", "pandas"]
    assert so.library_of(["python-3.x", "sklearn"]) == "scikit-learn"
    assert so.library_of(["pandas"]) is None  # not a python question
    assert so.license_for({}, date(2010, 1, 1)) == "CC BY-SA 2.5"
    assert so.license_for({}, date(2015, 1, 1)) == "CC BY-SA 3.0"


def test_html_to_text_fences_code():
    text = so.html_to_text("<p>Try <code>pd.concat</code>:</p><pre><code>a = 1\nb = 2\n</code></pre><p>done</p>")
    assert "`pd.concat`" in text and "```python\na = 1\nb = 2\n```" in text and text.endswith("done")


def test_iter_pairs_filters(tmp_path):
    pairs = list(so.iter_pairs(_posts_xml(tmp_path)))
    assert [(q["Id"], a["Id"], lib) for q, a, lib in pairs] == [("1", "3", "pandas")]


def test_make_chunk_with_inferred_version(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(versions, "_fetch_pypi", lambda p: {"0.23.4": "2018-08-03", "0.24.1": "2019-02-03",
                                                              "0.24.2": "2019-03-12", "1.0.0rc0": "2019-01-01"})
    versions._TIMELINES.clear()
    q, a, lib = next(so.iter_pairs(_posts_xml(tmp_path)))
    chunk = so.make_chunk(q, a, lib, versions.infer_version)
    assert chunk.version == "0.24.1" and chunk.version_inferred
    assert chunk.url == "https://stackoverflow.com/a/3" and chunk.author == "user 42" and chunk.score == 25
    assert chunk.license == "CC BY-SA 4.0" and "```python\ndf = pd.concat([df, row])\n```" in chunk.text


def test_infer_version_explicit_mention(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(versions, "_fetch_pypi", lambda p: {"1.0.0": "2020-01-29"})
    versions._TIMELINES.clear()
    assert versions.infer_version("pandas", "I am on pandas 0.25", date(2021, 1, 1)) == ("0.25", False)
    assert versions.infer_version("pandas", "no version here", date(2021, 1, 1)) == ("1.0.0", True)
    assert versions.infer_version("pandas", "no version here", date(2000, 1, 1)) == (None, True)
