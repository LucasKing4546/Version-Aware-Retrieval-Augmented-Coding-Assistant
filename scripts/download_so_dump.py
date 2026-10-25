"""Download the Stack Overflow data dump (resumable) into data/raw/stackoverflow/.

Usage: python scripts/download_so_dump.py [--url URL] [--out PATH]
Re-running continues a partial download. The default is the 2025-12-31 community release on archive.org
(stackoverflow.com.7z, ~69 GB). Extract only Posts.xml afterwards with 7-Zip: `7z e stackoverflow.com.7z Posts.xml`.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import typer
from tqdm import tqdm

DUMP_URL = (
    "https://archive.org/download/stackexchange_20251231/stackexchange_20251231/stackoverflow.com.7z"
)
DEFAULT_OUT = Path("data/raw/stackoverflow/stackoverflow.com.7z")
BLOCK = 1 << 20


def resume_headers(out: Path) -> dict[str, str]:
    """Range header continuing after the bytes already on disk (empty for a fresh download)."""
    done = out.stat().st_size if out.exists() else 0
    return {"Range": f"bytes={done}-"} if done else {}


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = resume_headers(out)
    start = out.stat().st_size if headers else 0
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as resp:
        if headers and resp.status != 206:  # server ignored the range: start over
            start = 0
        total = start + int(resp.headers.get("Content-Length", 0))
        mode = "ab" if start else "wb"
        with open(out, mode) as f, tqdm(total=total, initial=start, unit="B", unit_scale=True) as bar:
            while block := resp.read(BLOCK):
                f.write(block)
                bar.update(len(block))
    typer.echo(f"saved {out} ({out.stat().st_size / 1e9:.1f} GB)")


def main(url: str = DUMP_URL, out: Path = DEFAULT_OUT) -> None:
    free = shutil.disk_usage(out.parent if out.parent.exists() else Path(".")).free
    typer.echo(f"free disk: {free / 1e9:.0f} GB (need ~70 GB for the archive, ~150 GB more to extract Posts.xml)")
    download(url, out)


if __name__ == "__main__":
    typer.run(main)
