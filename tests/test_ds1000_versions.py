from snakepp.evaluation.ds1000 import LIBRARIES, pinned_versions, target_versions


def test_pinned_versions_cover_our_libraries():
    pins = pinned_versions()
    assert set(LIBRARIES.values()) == set(pins)
    assert pins["pandas"] == "1.5.3"


def test_pinned_versions_skips_comments(tmp_path):
    path = tmp_path / "versions.txt"
    path.write_text("# header\n\npandas==1.5.3  # inline\nnumpy\n", encoding="utf-8")
    assert pinned_versions(path) == {"pandas": "1.5.3"}


def test_target_versions_maps_ds1000_names():
    assert target_versions("Sklearn") == {"scikit-learn": "1.4.0"}
