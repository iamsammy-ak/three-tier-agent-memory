"""Episodic compression: a conversation segment -> the few sentences worth keeping.

``ExtractiveCompressor`` is the offline method: salient sentences (high importance) are always kept verbatim, then
an importance-biased TextRank (Mihalcea and Tarau, 2004) fills the remaining token budget with the most central
sentences. TextRank builds a graph whose nodes are sentences and whose edges are word overlap, then runs PageRank
on it; the personalisation vector is the sentence importance, so central *and* important sentences win.
``LLMCompressor`` adds an abstractive summary from a model while still keeping salient sentences verbatim, so
provenance and fact links survive.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from .llm import LLM, LLMError
from .models import EpisodeSentence
from .tokens import ApproxTokenCounter, TokenCounter

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“(])")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the and or but of to in on at for with by from is are was were be been am i me my we our you your it its "
    "this that these those do does did so as just very really have has had will would can could should there here "
    "what when where which who how not no yes ok okay oh hi hello hey thanks".split()
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation followed by a capital, digit or quote."""
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def _content_words(sentence: str) -> set[str]:
    words = set()
    for w in _WORD.findall(sentence.lower()):
        if w in _STOP or len(w) < 2:
            continue
        words.add(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w)
    return words


def textrank(
    sentences: Sequence[str],
    bias: Sequence[float] | None = None,
    *,
    damping: float = 0.85,
    iterations: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Personalised PageRank over the sentence-similarity graph. Returns scores that sum to 1."""
    n = len(sentences)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    words = [_content_words(s) for s in sentences]
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            overlap = len(words[i] & words[j])
            if overlap:
                w = overlap / (math.log(len(words[i]) + 1) + math.log(len(words[j]) + 1))
                weights[i, j] = weights[j, i] = w
    teleport = np.asarray(bias if bias is not None else np.ones(n), dtype=float) + 1e-9
    teleport = teleport / teleport.sum()
    row_sums = weights.sum(axis=1)
    dangling = row_sums == 0
    transition = np.divide(weights, row_sums[:, None], out=np.zeros_like(weights), where=~dangling[:, None])
    rank = np.full(n, 1.0 / n)
    for _ in range(iterations):
        leaked = rank[dangling].sum()
        updated = (1 - damping) * teleport + damping * (transition.T @ rank + leaked * teleport)
        if np.abs(updated - rank).sum() < tol:
            rank = updated
            break
        rank = updated
    return rank / rank.sum()


class CompressionResult(BaseModel):
    """Which sentences to keep verbatim, plus an optional abstractive summary."""

    keep: list[bool]
    abstract: str | None = None


class Compressor(Protocol):
    name: str

    def compress(self, sentences: Sequence[EpisodeSentence]) -> CompressionResult: ...


class ExtractiveCompressor:
    """Offline compression: salient sentences + importance-biased TextRank up to a token target."""

    name = "textrank"

    def __init__(
        self,
        counter: TokenCounter | None = None,
        *,
        ratio: float = 0.35,
        max_tokens: int = 160,
        min_tokens: int = 24,
        min_importance: float = 0.1,
    ) -> None:
        self.counter = counter or ApproxTokenCounter()
        self.ratio = ratio
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.min_importance = min_importance

    def target_tokens(self, raw_tokens: int) -> int:
        return int(min(self.max_tokens, max(self.min_tokens, self.ratio * raw_tokens)))

    def compress(self, sentences: Sequence[EpisodeSentence]) -> CompressionResult:
        n = len(sentences)
        if n == 0:
            return CompressionResult(keep=[])
        sizes = [self.counter(s.text) for s in sentences]
        keep = [s.salient for s in sentences]
        used = sum(size for size, k in zip(sizes, keep, strict=True) if k)
        target = self.target_tokens(sum(sizes))
        scores = textrank([s.text for s in sentences], [0.05 + s.importance for s in sentences])
        for idx in np.argsort(-scores, kind="stable"):
            i = int(idx)
            if keep[i] or sentences[i].importance < self.min_importance:
                continue
            if used + sizes[i] <= target:
                keep[i] = True
                used += sizes[i]
        if not any(keep):
            keep[int(np.argmax(scores))] = True
        return CompressionResult(keep=keep)


class _Summary(BaseModel):
    summary: str


_SUMMARY_PROMPT = """Summarise this conversation segment between a user and an assistant in at most 60 words.
Keep names, places, dates, numbers, preferences and decisions; drop greetings and small talk.
Return {"summary": "..."}."""


class LLMCompressor:
    """Abstractive summary from an LLM; salient sentences stay verbatim. Falls back to extractive on failure."""

    def __init__(self, llm: LLM, fallback: ExtractiveCompressor | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or ExtractiveCompressor()
        self.name = f"llm:{llm.name}"

    def compress(self, sentences: Sequence[EpisodeSentence]) -> CompressionResult:
        if not sentences:
            return CompressionResult(keep=[])
        transcript = "\n".join(f"{s.role}: {s.text}" for s in sentences)
        try:
            reply = self.llm.complete_json(
                [{"role": "system", "content": _SUMMARY_PROMPT}, {"role": "user", "content": transcript}], _Summary
            )
        except LLMError:
            return self.fallback.compress(sentences)
        return CompressionResult(keep=[s.salient for s in sentences], abstract=reply.summary.strip() or None)
