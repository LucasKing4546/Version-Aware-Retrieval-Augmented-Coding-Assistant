# Task: Start Stack Overflow data dump download; check watermark list (#3)

Owner: Lucas (prepared by Kevin with Claude Code) · Milestone: M0 · Due: 2026-10-26 · Branch: `feat/data-so-dump`

## Which dump
- **Chosen:** the **2025-12-31 community release** on archive.org: `stackoverflow.com.7z`, 68.96 GB, one archive holding all the tables.
  - URL: <https://archive.org/download/stackexchange_20251231/stackexchange_20251231/stackoverflow.com.7z>
  - A torrent is also available on the [item page](https://archive.org/details/stackexchange_20251231).
- **Why a community mirror:** Stack Exchange stopped its own archive.org release.
- **License:** CC BY-SA. Every row carries its own `ContentLicense`.
- **Checksums:** listed at <https://github.com/LunarWatcher/se-data-dump-transformer/blob/master/docs/meta/Hashes.md>.

## Watermarks (checked 2026-10-25)
Source: the [community list of known watermarks](https://github.com/LunarWatcher/se-data-dump-transformer/blob/master/docs/meta/Known%20watermarks.md) and the dump README.

| Dump | Status |
|---|---|
| ≤ 2025-03-30 | clean |
| 2025-06-30 (rev1, rev2) | **watermarked ("v1")**: fabricated rows in `Posts.xml` only |
| 2025-09-30 | none identified, but considered likely |
| 2025-12-31 (ours) | none identified when the README was written |

**How to spot the v1 rows:**
- **Ids:** always `Id="1000000001"` (a question) and `Id="1000000010"` (an answer).
- **Other fields:** `OwnerUserId="-1"`, `Score="1"`, `CommentCount="0"`, `CreationDate` on 2025-06-01.
- **Not in history:** the rows are missing from `PostHistory.xml`.

**The filter we added:** `snakepp.corpus.stackoverflow.is_watermark(row)`. It drops the two known ids, plus any post with an id of 1e9 or more owned by user -1. Real Stack Overflow ids are around 8e7, so this can't catch real posts.
The Posts.xml parser must call it on every row. Recheck the list before freezing the corpus (M3).

## What was done
- **`scripts/download_so_dump.py`** is a typer CLI. It saves to `data/raw/stackoverflow/` (git-ignored) and shows a progress bar.
  If the download stops, running it again continues from where it stopped. I tested this, including the resume, on a small file from the same server.
- **`snakepp/corpus/stackoverflow.py`** now has `is_watermark()`. The rest of the parser is still Lucas's M1 task.
- **Tests:** `tests/test_so_dump.py` has 4, covering the watermark filter and the resume header.

## Still to do (Lucas)
1. **Start the download.** I didn't start it: it's 69 GB and should run on the machine that will do the parsing.
   ```bash
   python scripts/download_so_dump.py
   ```
   Disk space: about 70 GB for the archive, plus roughly 150 GB more for `Posts.xml`.
2. **Extract only the posts.** Install 7-Zip, then run `7z e data/raw/stackoverflow/stackoverflow.com.7z Posts.xml -odata/raw/stackoverflow/`.
3. **Verify the archive.** Compare its SHA against the hash list before parsing.
