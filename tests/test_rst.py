from snakepp.corpus.rst import Section, clean_rst, pack, split_sections

RST = """\
.. _whatsnew_200:

=====================
What's new in 2.0.0
=====================

Intro text about the release.

Removal of prior version deprecations
-------------------------------------

Removed :meth:`DataFrame.append`, use :func:`~pandas.concat` instead (:issue:`35407`).

.. code-block:: python

    df = pd.concat([df, other])

    # still part of the block

Other
~~~~~

Tiny.
"""


def test_clean_rst_roles_and_targets():
    text = clean_rst(RST)
    assert "DataFrame.append" in text and "pandas.concat" in text and ":meth:" not in text
    assert ".. _whatsnew_200:" not in text and ".. code-block:: python" in text


def test_split_sections_heading_path():
    secs = split_sections(clean_rst(RST), title="whatsnew/v2.0.0")
    assert [s.path for s in secs] == [
        ("whatsnew/v2.0.0", "What's new in 2.0.0"),
        ("whatsnew/v2.0.0", "What's new in 2.0.0", "Removal of prior version deprecations"),
        ("whatsnew/v2.0.0", "What's new in 2.0.0", "Removal of prior version deprecations", "Other"),
    ]
    assert "still part of the block" in secs[1].text


def test_pack_merges_tiny_and_keeps_code_blocks():
    secs = pack(split_sections(clean_rst(RST)), max_chars=3000, min_chars=200)
    assert len(secs) == 1 and "Other" in secs[0].text and "Tiny." in secs[0].text


def test_pack_splits_long_sections_outside_code():
    para = "word " * 100
    code = "Example::\n\n    x = 1\n\n    y = 2"
    text = "\n\n".join([para, para, code, para])
    out = pack([Section(("a",), text)], max_chars=1100, min_chars=10)
    assert len(out) >= 2
    assert any("Example::" in s.text and "y = 2" in s.text for s in out)  # block kept with its intro
