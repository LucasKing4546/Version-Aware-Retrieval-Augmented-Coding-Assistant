"""stackoverflow — owner: Lucas, due: M1 Wed 28 Oct.

TODO:
Stream Posts.xml with lxml iterparse (flat memory). Keep python questions tagged with one of our libraries,
with an accepted answer and score >= 3. One Chunk per question + accepted answer; HTML to text but keep
<code> as fenced code. Fill url, score, created, author and license-by-date. Filter known watermark rows.
"""

from __future__ import annotations
