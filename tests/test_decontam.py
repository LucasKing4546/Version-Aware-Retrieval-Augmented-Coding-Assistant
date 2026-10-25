import csv

import numpy as np

from snakepp.corpus import decontam
from snakepp.corpus.io import read_chunks
from snakepp.schemas import Task

PROMPT = ("I have a dataframe with columns user, date and value and I want to fill the missing dates for every "
          "user with zero values so that each user has a continuous range of dates between min and max")


def _task(text, tid="ds1000/1"):
    return Task(task_id=tid, benchmark="ds1000", prompt=text)


def test_shingles():
    assert decontam.shingles("a b c d e f") == {"a b c d e", "b c d e f"}
    assert decontam.shingles("Hi there") == {"hi there"}
    assert decontam.shingles("") == set()


def test_find_matches_flags_near_duplicate(chunks_200):
    leaked = chunks_200[150].model_copy(update={"text": "Q: fill dates\n\n" + PROMPT + " thanks!"})
    chunks = [*chunks_200[:150], leaked]
    matches = decontam.find_matches([_task(PROMPT), _task("completely unrelated words here today", "x")], chunks)
    assert [(m.task_id, m.chunk_id) for m in matches] == [("ds1000/1", leaked.id)]
    assert matches[0].jaccard > 0.8


def test_confirm_with_cosine(chunks_200):
    m = decontam.Match("t", chunks_200[0].id, 0.5)
    same = lambda texts: np.ones((len(texts), 4)) / 2  # noqa: E731 - cosine 1.0
    assert decontam.confirm_with_cosine([m], {"t": _task("x", "t")}, {m.chunk_id: chunks_200[0]}, same, 0.9)[0].cosine
    zero = lambda texts: np.zeros((len(texts), 4))  # noqa: E731
    assert decontam.confirm_with_cosine([m], {"t": _task("x", "t")}, {m.chunk_id: chunks_200[0]}, zero, 0.9) == []


def test_write_outputs(tmp_path, chunks_200):
    so = [c for c in chunks_200 if c.source == "stackoverflow"]
    matches = [decontam.Match("ds1000/1", so[0].id, 0.9)]
    clean, leaky = decontam.write_outputs(matches, so, tmp_path, tmp_path)
    assert (clean, leaky) == (len(so) - 1, len(so))
    assert so[0].id not in {c.id for c in read_chunks(tmp_path / "so_clean.parquet")}
    rows = list(csv.DictReader(open(tmp_path / "matches.csv", encoding="utf-8")))
    assert rows[0]["chunk_id"] == so[0].id
