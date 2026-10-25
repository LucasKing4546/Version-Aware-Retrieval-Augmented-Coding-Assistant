import importlib.util
import sys
from pathlib import Path

from snakepp.corpus.stackoverflow import is_watermark


def _script():
    spec = importlib.util.spec_from_file_location("download_so_dump", Path("scripts/download_so_dump.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_watermark_v1_ids():
    assert is_watermark({"Id": "1000000001", "OwnerUserId": "-1"})
    assert is_watermark({"Id": "1000000010", "PostTypeId": "2"})


def test_watermark_high_id_community_owner():
    assert is_watermark({"Id": "1000000123", "OwnerUserId": "-1"})
    assert not is_watermark({"Id": "1000000123", "OwnerUserId": "42"})


def test_real_posts_kept():
    assert not is_watermark({"Id": "11227809", "OwnerUserId": "-1"})  # community-wiki style post
    assert not is_watermark({"Id": "79000000", "OwnerUserId": "123"})
    assert not is_watermark({})


def test_resume_headers(tmp_path):
    out = tmp_path / "dump.7z"
    script = _script()
    assert script.resume_headers(out) == {}
    out.write_bytes(b"x" * 10)
    assert script.resume_headers(out) == {"Range": "bytes=10-"}
