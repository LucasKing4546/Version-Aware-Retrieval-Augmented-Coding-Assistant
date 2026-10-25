"""docs — owner: Lucas, due: M1 Tue 27 Oct.

TODO:
Parse a Sphinx docs tree (.rst; .md for Hugging Face) for one library at one version. Split by section heading,
never split a code block, merge tiny sections (~300-800 tokens per chunk), yield Chunk(source='docs') with
stable ids, and write data/chunks/docs_<library>_<version>.parquet via a typer CLI.
"""

from __future__ import annotations
