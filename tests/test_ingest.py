import json
from datetime import date

from snakepp.ingest import so_fetch
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

