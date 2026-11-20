"""Sentence embeddings: fastembed ``BAAI/bge-small-en-v1.5`` when available, a hashing embedder otherwise.

Both return L2-normalised float32 rows, so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOP = frozenset(
    "a an the and or but of to in on at for with by from is are was were be been am i me my we our you your "
    "it its this that these those do does did so as just very really user user's".split()
)


class Embedder(Protocol):
    """Anything that turns texts into L2-normalised vectors."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), float32, rows L2-normalised (zero rows for empty text)."""
        ...


def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


@lru_cache(maxsize=200_000)
def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, (1.0 if (value >> 63) & 1 else -1.0)


class HashingEmbedder:
    """Deterministic signed feature hashing of words, word bigrams and character n-grams.

    No model download and stable across processes (BLAKE2 rather than Python's salted ``hash``). It captures
    lexical overlap only, so it is a stand-in for tests and offline runs, not a semantic model.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
        feats: list[tuple[str, float]] = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.6) for a, b in zip(words, words[1:], strict=False)]
        for w in words:
            padded = f"<{w}>"
            for n in (3, 4):
                feats += [(f"c:{padded[i : i + n]}", 0.25) for i in range(max(1, len(padded) - n + 1))]
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                idx, sign = _bucket(feature, self.dim)
                out[row, idx] += sign * weight
        return normalise_rows(out)


class FastEmbedEmbedder:
    """fastembed ONNX model (``BAAI/bge-small-en-v1.5``, 384-d by default); honours ``FASTEMBED_CACHE_PATH``."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding

        cache = cache_dir or os.environ.get("FASTEMBED_CACHE_PATH")
        kwargs: dict[str, Any] = {"model_name": model_name}
        if cache:
            kwargs["cache_dir"] = cache
        self._model = TextEmbedding(**kwargs)
        self.name = f"fastembed:{model_name}"
        self.dim = int(self.embed(["probe"]).shape[1])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(self, "dim", 384)), dtype=np.float32)
        safe = [t if t.strip() else " " for t in texts]
        vectors = np.asarray(list(self._model.embed(safe, batch_size=64)), dtype=np.float32)
        return normalise_rows(vectors)


def get_embedder(kind: str = "auto") -> Embedder:
    """``hashing``, ``fastembed`` or ``auto`` (fastembed when installed and loadable, hashing otherwise)."""
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder()
    if kind == "auto":
        try:
            return FastEmbedEmbedder()
        except Exception:  # missing package, no network for the first download, ONNX runtime issues
            return HashingEmbedder()
    raise ValueError(f"unknown embedder {kind!r} (use 'auto', 'fastembed' or 'hashing')")
