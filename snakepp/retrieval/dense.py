"""dense — owner: Kevin, due: M1 Thu 29 Oct.

TODO:
sentence-transformers embeddings (model from config), normalized, FAISS IndexFlatIP with id map, embeddings
cached as .npy keyed by model + chunk-file hash. search(query, k) -> list[Retrieved(stage='dense')].
"""

from __future__ import annotations
