"""HashingEncoder: a deterministic stand-in for the sentence-transformers model, so dense tests need no download.

Bag of tokens hashed into `dim` buckets, L2-normalised: texts sharing words get a high cosine. It also counts
how many texts it has encoded, which the embedding-cache tests check.
"""

from __future__ import annotations

import hashlib

import numpy as np

from snakepp.retrieval.tokenizer import tokenize


class HashingEncoder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.encoded = 0

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.encoded += len(texts)
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)
