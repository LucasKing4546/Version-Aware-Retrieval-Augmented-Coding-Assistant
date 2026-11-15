"""Version resolver and the soft/hard version filter, on hand-made chunks and on the real fixture."""

from pathlib import Path

import pytest

from snakepp.config import Config, RetrievalConfig, VersionFilterConfig
from snakepp.ingest.chunks import load_chunks
from snakepp.retrieval import version_filter
from snakepp.retrieval.retriever import Retriever
from snakepp.retrieval.versions import compatibility, detect_versions
from snakepp.schemas import Chunk
from tests.fakes.fake_encoder import HashingEncoder

FIXTURE = Path(__file__).parent / "fixtures" / "fixture_chunks.jsonl"


def chunk(cid: str, version: str | None, inferred: bool = False, library: str = "pandas") -> Chunk:
    source = "stackoverflow" if inferred else "docs"
    return Chunk(id=cid, text=cid, source=source, library=library, version=version, version_inferred=inferred,
                 url=f"https://x/{cid}", title=cid, license="BSD-3-Clause")  # fmt: skip


# ---- detect_versions -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("pandas==1.5.3\nnumpy==1.26.4\n", {"pandas": "1.5.3", "numpy": "1.26.4"}),
        ("I'm on pandas 2.2.3 and Python 3.11", {"pandas": "2.2.3", "python": "3.11"}),
        ("with pandas version 1.5, how do I append a row?", {"pandas": "1.5"}),
        ("sklearn~=1.4 and scikit-learn 1.5.0", {"scikit-learn": "1.4"}),  # the pin wins over prose
        ("python3.12 -m pip install scipy===1.13.1", {"python": "3.12", "scipy": "1.13.1"}),
        ("pandas>=2.0,<3 and numpy!=2.0.0", {}),  # ranges are not a target
        ("How do I append a row to a DataFrame in pandas 2?", {}),  # no dotted version, too vague
        ("geopandas==0.14.0", {}),  # not one of our libraries
    ],
)
def test_detect_versions(text, expected):
    assert detect_versions(text) == expected


# ---- compatibility ---------------------------------------------------------------------------------------


def test_compatibility_is_by_major_version():
    target = {"pandas": "2.2.3"}
    assert compatibility(chunk("a", "2.0.1"), target) == "match"
    assert compatibility(chunk("b", "1.5.3"), target) == "mismatch"
    assert compatibility(chunk("c", "0.18.0", inferred=True), target) == "mismatch"
    assert compatibility(chunk("d", None), target) == "unknown"
    assert compatibility(chunk("e", "1.26.4", library="numpy"), target) == "unknown"  # no numpy target
    assert compatibility(chunk("f", "1.5.3"), {}) == "unknown"


# ---- version_filter.apply --------------------------------------------------------------------------------

CHUNKS = {c.id: c for c in [chunk("new", "2.2.3"), chunk("old", "1.5.3"), chunk("so_old", "0.18.0", True),
                            chunk("nover", None)]}  # fmt: skip
HITS = [("old", 1.0), ("so_old", 0.9), ("new", 0.5), ("nover", 0.4)]
TARGET = {"pandas": "2.2.3"}


def run(mode: str, penalty: float = 0.3):
    return version_filter.apply(HITS, CHUNKS, TARGET, VersionFilterConfig(mode=mode, penalty=penalty))


def test_off_changes_nothing():
    hits, decisions = run("off")
    assert hits == HITS and {d.action for d in decisions.values()} == {"kept"}


def test_soft_penalises_and_reorders_with_a_softer_penalty_for_inferred_versions():
    hits, decisions = run("soft", penalty=0.3)
    assert hits == [("so_old", pytest.approx(0.9 * 0.65)), ("new", 0.5), ("nover", 0.4), ("old", pytest.approx(0.3))]
    assert decisions["old"].factor == 0.3 and decisions["so_old"].factor == pytest.approx(0.65)
    assert decisions["new"].action == "kept" and decisions["nover"].compat == "unknown"


def test_hard_drops_mismatches_only():
    hits, decisions = run("hard")
    assert hits == [("new", 0.5), ("nover", 0.4)]
    assert decisions["old"].action == decisions["so_old"].action == "dropped"


def test_negative_scores_are_never_raised():
    hits, _ = version_filter.apply([("old", -0.2)], CHUNKS, TARGET, VersionFilterConfig(mode="soft", penalty=0.5))
    assert hits[0][1] < -0.2


# ---- in the Retriever, on the real fixture ---------------------------------------------------------------


@pytest.fixture(scope="module")
def real_chunks() -> list[Chunk]:
    return list(load_chunks(FIXTURE))


def retriever(chunks, tmp_path, mode: str, penalty: float = 0.3) -> Retriever:
    cfg = Config(name=mode, retrieval=RetrievalConfig(mode="bm25", rerank=False),
                 version_filter=VersionFilterConfig(mode=mode, penalty=penalty))  # fmt: skip
    return Retriever(chunks, cfg, encode=HashingEncoder(), cache_dir=tmp_path)


def test_hard_filter_keeps_only_target_major_on_the_append_trap(real_chunks, tmp_path):
    query = "DataFrame.append iteritems applymap"
    off = retriever(real_chunks, tmp_path, "off").retrieve(query, "pandas", {"pandas": "2.2.3"}, k=10)
    hard = retriever(real_chunks, tmp_path, "hard").retrieve(query, "pandas", {"pandas": "2.2.3"}, k=10)
    assert any(h.chunk.version == "1.5.3" for h in off)
    assert hard and all(h.chunk.version is None or h.chunk.version.startswith("2.") for h in hard)
    assert [h.rank for h in hard] == list(range(1, len(hard) + 1))


def test_soft_filter_moves_target_version_up(real_chunks, tmp_path):
    query, target = "DataFrame.applymap", {"pandas": "2.2.3"}
    off = retriever(real_chunks, tmp_path, "off").retrieve(query, "pandas", target, k=10)
    soft = retriever(real_chunks, tmp_path, "soft", penalty=0.1).retrieve(query, "pandas", target, k=10)

    def share_new(hits):
        return sum(h.chunk.version == "2.2.3" for h in hits[:5])

    assert share_new(soft) > share_new(off)
    assert soft[0].chunk.version == "2.2.3"
