from snakepp.retrieval.fusion import rrf
from snakepp.retrieval.version_filter import apply, compatible
from snakepp.schemas import Chunk, Retrieved


def _r(cid, rank, score=1.0, stage="bm25"):
    return Retrieved(chunk_id=cid, rank=rank, score=score, stage=stage)


def _c(cid, version):
    return Chunk(id=cid, text="", source="docs", library="pandas", version=version, url="u", title=cid, license="BSD")


def test_rrf_prefers_items_in_both_lists():
    fused = rrf([[_r("a", 1), _r("b", 2)], [_r("b", 1), _r("c", 2)]])
    assert fused[0].chunk_id == "b" and fused[0].stage == "fused"


def test_version_filter_modes():
    chunks = {"old": _c("old", "1.3.0"), "new": _c("new", "2.2.0")}
    res = [_r("old", 1, 0.9), _r("new", 2, 0.8)]
    target = {"pandas": "2.2.2"}
    assert not compatible(chunks["old"], target)
    assert [r.chunk_id for r in apply(res, chunks, target, "hard")] == ["new"]
    soft = apply(res, chunks, target, "soft", penalty=0.3)
    assert [r.chunk_id for r in soft] == ["new", "old"]
    assert apply(res, chunks, target, "off") == res
