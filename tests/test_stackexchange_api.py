import pytest

from snakepp import cache
from snakepp.corpus import stackexchange_api as se
from snakepp.corpus.stackoverflow import make_chunk


def _q(qid, score, aid):
    return {"question_id": qid, "title": "Add &#39;row&#39;", "body": "<p>q</p>", "score": score,
            "tags": ["python", "pandas"], "accepted_answer_id": aid, "creation_date": 1_600_000_000}


def _a(aid):
    return {"answer_id": aid, "body": "<pre><code>pd.concat(x)</code></pre>", "score": 7,
            "owner": {"user_id": 9, "display_name": "Ana &amp; Bo"}, "content_license": "CC BY-SA 4.0",
            "creation_date": 1_600_000_100}


class FakeAPI:
    def __init__(self):
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        if "/answers/" in url:
            ids = url.split("/answers/")[1].split("?")[0].replace("%3B", ";").split(";")
            return {"items": [_a(int(i)) for i in ids]}
        return {"items": [_q(1, 50, 11), _q(2, 9, 12), _q(3, 2, 13)], "has_more": True}


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("SE_API_KEY", raising=False)
    monkeypatch.setattr(se, "load_dotenv", lambda: None)
    monkeypatch.setattr(se, "YEARS", range(2020, 2021))


def test_top_questions_stops_below_min_score():
    api = FakeAPI()
    qs = se.top_questions("pandas", 2020, 10, 3, api)
    assert [q["question_id"] for q in qs] == [1, 2] and len(api.urls) == 1
    assert "tagged=python%3Bpandas" in api.urls[0] and "accepted=True" in api.urls[0]


def test_pairs_and_cached_calls():
    api = FakeAPI()
    pairs = list(se.iter_pairs("pandas", per_library=5, fetch=api))
    assert [(q["Id"], a["Id"]) for q, a in pairs] == [("1", "11"), ("2", "12")]
    calls = len(api.urls)
    list(se.iter_pairs("pandas", per_library=5, fetch=api))
    assert len(api.urls) == calls  # second run served from data/cache/seapi


def test_rows_feed_make_chunk():
    q, a = se.question_row(_q(1, 50, 11)), se.answer_row(_a(11))
    chunk = make_chunk(q, a, "pandas")
    assert chunk.title == "Add 'row'" and chunk.author == "Ana & Bo" and chunk.license == "CC BY-SA 4.0"
    assert chunk.url == "https://stackoverflow.com/a/11" and "```python\npd.concat(x)\n```" in chunk.text


def test_quota_error():
    with pytest.raises(se.QuotaExceeded, match="SE_API_KEY"):
        se.get("search/advanced", {"page": 1}, lambda url: {"error_id": 502, "error_name": "throttle_violation"})
