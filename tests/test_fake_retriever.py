import inspect

import pytest

from snakepp.config import Config, load_config
from snakepp.retrieval.retriever import Retriever
from tests.fakes.fake_retriever import FakeRetriever, tokenize


@pytest.fixture(scope="module")
def fake() -> FakeRetriever:
    return FakeRetriever.from_config(Config(name="test"))


def test_same_signature_as_real_retriever():
    for name in ("from_config", "retrieve"):
        real = list(inspect.signature(getattr(Retriever, name)).parameters)
        assert list(inspect.signature(getattr(FakeRetriever, name)).parameters)[: len(real)] == real


def test_tokenize_splits_identifiers_and_drops_stopwords():
    assert tokenize("How do I use DataFrame.append?") == {"dataframe", "append"}


def test_results_are_ranked_and_filled(fake):
    hits = fake.retrieve("append rows to a DataFrame", "pandas", {"pandas": "2.2.2"}, k=5)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(h.chunk is not None and h.chunk.id == h.chunk_id for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_library_filter(fake):
    hits = fake.retrieve("concatenate arrays", "numpy", {}, k=10)
    assert hits and all(h.chunk.library == "numpy" for h in hits)


def test_version_trap_ties_prefer_target_version(fake):
    old = fake.retrieve("pandas concat", "pandas", {"pandas": "1.5.3"}, k=1)[0].chunk
    new = fake.retrieve("pandas concat", "pandas", {"pandas": "2.2.2"}, k=1)[0].chunk
    assert (old.version, new.version) == ("1.5.3", "2.2.2")


def test_append_question_finds_the_removed_api(fake):
    texts = [h.chunk.text for h in fake.retrieve("DataFrame.append", "pandas", {"pandas": "2.2.2"}, k=10)]
    assert any("df.append" in t for t in texts) and any("concat" in t for t in texts)


def test_sources_from_config():
    docs_only = FakeRetriever.from_config(load_config("c1_docs"))
    assert all(h.chunk.source == "docs" for h in docs_only.retrieve("append", None, {}, k=20))
    assert FakeRetriever.from_config(load_config("c0_none")).retrieve("append", None, {}, k=5) == []


def test_no_match_and_get_chunk(fake):
    assert fake.retrieve("zzzz qqqq", None, {}, k=3) == []
    hit = fake.retrieve("append", None, {}, k=1)[0]
    assert fake.get_chunk(hit.chunk_id) == hit.chunk
    with pytest.raises(KeyError):
        fake.get_chunk("missing")


def test_deterministic(fake):
    assert fake.retrieve("groupby mean", "pandas", {}, k=5) == fake.retrieve("groupby mean", "pandas", {}, k=5)
