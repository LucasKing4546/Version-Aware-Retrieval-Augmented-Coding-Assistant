"""stackoverflow — owner: Lucas, due: M1 Wed 28 Oct.

TODO:
Stream Posts.xml with lxml iterparse (flat memory). Keep python questions tagged with one of our libraries,
with an accepted answer and score >= 3. One Chunk per question + accepted answer; HTML to text but keep
<code> as fenced code. Fill url, score, created, author and license-by-date. Filter known watermark rows.
"""

from __future__ import annotations

from collections.abc import Mapping

# Fabricated rows Stack Exchange planted in the 2025-06-30 dump ("v1" watermark), see
# https://github.com/LunarWatcher/se-data-dump-transformer/blob/master/docs/meta/Known%20watermarks.md
WATERMARK_V1_IDS = {"1000000001", "1000000010"}
# Real Stack Overflow post ids are far below this; anything above is fabricated.
MAX_REAL_POST_ID = 1_000_000_000


def is_watermark(row: Mapping[str, str]) -> bool:
    """True for a Posts.xml row (its XML attributes) that matches a known watermark pattern."""
    post_id = row.get("Id", "")
    if post_id in WATERMARK_V1_IDS:
        return True
    return post_id.isdigit() and int(post_id) >= MAX_REAL_POST_ID and row.get("OwnerUserId") == "-1"
