"""decontam — owner: Lucas, due: M3 Wed 4 Nov.

TODO:
Find near-duplicates between benchmark prompts and SO chunks with datasketch MinHash LSH (5-word shingles),
confirm with embedding cosine. Write a matches CSV and clean + leaky parquet variants. Thresholds are CLI options.
"""

from __future__ import annotations
