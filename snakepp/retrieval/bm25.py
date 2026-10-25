"""bm25 — owner: Kevin, due: M1 Wed 28 Oct.

TODO:
rank_bm25 index over chunk parquet files. Identifier-aware tokenizer: lowercase, keep whole dotted identifiers
AND split on '.', '_', parentheses and camelCase. search(query, k) -> list[Retrieved(stage='bm25')].
"""

from __future__ import annotations
