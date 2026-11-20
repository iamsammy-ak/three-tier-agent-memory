"""Memory backends compared in the benchmark.

* ``RecencyWindow``: keep the full transcript and hand the agent its most recent turns (what a plain chat loop does
  when the history is truncated to fit the context window).
* ``FlatVectorStore``: embed every turn and return the nearest neighbours of the question; no tiers, no importance,
  no supersession, no eviction.
* ``TieredBackend``: this library, optionally with supersession or eviction switched off (ablations).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..clock import ManualClock
from ..config import MemoryConfig
from ..embeddings import Embedder
from ..memory import MemorySystem
from ..models import ConsolidationReport
from ..store import SQLiteStore
from ..tokens import ApproxTokenCounter, TokenCounter


class Backend(Protocol):
    name: str

    def observe(self, user_id: str, session: int, role: str, text: str, ts: float) -> None: ...

    def end_session(self, user_id: str, session: int, ts: float) -> None: ...

    def finish(self, user_id: str, ts: float) -> None: ...

    def query(self, user_id: str, question: str, *, k: int | None, budget_tokens: int) -> list[str]: ...

    def size(self, user_id: str) -> tuple[int, int]: ...


def _pack(texts: list[str], order: list[int], k: int | None, budget: int, counter: TokenCounter) -> list[str]:
    out: list[str] = []
    used = 0
    for i in order:
        if k is not None and len(out) >= k:
            break
        tokens = counter(texts[i])
        if used + tokens > budget:
            continue
        used += tokens
        out.append(texts[i])
    return out


class RecencyWindow:
    """Full history, truncated to the most recent turns at query time."""

    name = "recency-window"

    def __init__(self, counter: TokenCounter | None = None) -> None:
        self.counter = counter or ApproxTokenCounter()
        self.history: dict[str, list[str]] = {}

    def observe(self, user_id: str, session: int, role: str, text: str, ts: float) -> None:
        self.history.setdefault(user_id, []).append(text)

    def end_session(self, user_id: str, session: int, ts: float) -> None:
        return None

    def finish(self, user_id: str, ts: float) -> None:
        return None

    def query(self, user_id: str, question: str, *, k: int | None, budget_tokens: int) -> list[str]:
        texts = self.history.get(user_id, [])
        order = list(range(len(texts) - 1, -1, -1))
        packed = _pack(texts, order, k, budget_tokens, self.counter)
        return packed[::-1]

    def size(self, user_id: str) -> tuple[int, int]:
        texts = self.history.get(user_id, [])
        return len(texts), sum(len(t.encode("utf-8")) for t in texts)


class FlatVectorStore:
    """Every turn embedded and searched by cosine similarity."""

    name = "flat-vector"

    def __init__(self, embedder: Embedder, counter: TokenCounter | None = None) -> None:
        self.embedder = embedder
        self.counter = counter or ApproxTokenCounter()
        self.texts: dict[str, list[str]] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self._pending: dict[str, list[str]] = {}

    def observe(self, user_id: str, session: int, role: str, text: str, ts: float) -> None:
        self._pending.setdefault(user_id, []).append(text)

    def end_session(self, user_id: str, session: int, ts: float) -> None:
        pending = self._pending.pop(user_id, [])
        if not pending:
            return
        new = self.embedder.embed(pending)
        old = self.vectors.get(user_id)
        self.vectors[user_id] = new if old is None else np.vstack([old, new])
        self.texts.setdefault(user_id, []).extend(pending)

    def finish(self, user_id: str, ts: float) -> None:
        return None

    def query(self, user_id: str, question: str, *, k: int | None, budget_tokens: int) -> list[str]:
        texts = self.texts.get(user_id, [])
        if not texts:
            return []
        scores = self.vectors[user_id] @ self.embedder.embed([question])[0]
        order = [int(i) for i in np.argsort(-scores, kind="stable")]
        return _pack(texts, order, k, budget_tokens, self.counter)

    def size(self, user_id: str) -> tuple[int, int]:
        texts = self.texts.get(user_id, [])
        vectors = self.vectors.get(user_id)
        return len(texts), sum(len(t.encode("utf-8")) for t in texts) + (0 if vectors is None else vectors.nbytes)


class TieredBackend:
    """The three-tier memory, one namespace per simulated user, offline components."""

    def __init__(self, name: str, embedder: Embedder, config: MemoryConfig) -> None:
        self.name = name
        self.clock = ManualClock()
        self.store = SQLiteStore()
        self.base = MemorySystem(self.store, namespace="bench", config=config, embedder=embedder, clock=self.clock,
                                 offline=True)
        self.systems: dict[str, MemorySystem] = {}
        self.reports: list[ConsolidationReport] = []

    def _mem(self, user_id: str) -> MemorySystem:
        if user_id not in self.systems:
            self.systems[user_id] = self.base.for_namespace(user_id)
        return self.systems[user_id]

    def observe(self, user_id: str, session: int, role: str, text: str, ts: float) -> None:
        self.clock.set(ts)
        result = self._mem(user_id).observe(f"s{session}", role, text)
        if result.consolidation is not None:
            self.reports.append(result.consolidation)

    def end_session(self, user_id: str, session: int, ts: float) -> None:
        self.clock.set(ts)
        self.reports.append(self._mem(user_id).end_session(f"s{session}"))

    def finish(self, user_id: str, ts: float) -> None:
        self.clock.set(ts)
        self._mem(user_id).maintain()

    def query(self, user_id: str, question: str, *, k: int | None, budget_tokens: int) -> list[str]:
        result = self._mem(user_id).recall(question, budget_tokens=budget_tokens, k=k, reinforce=False)
        return [item.text for item in result.items]

    def size(self, user_id: str) -> tuple[int, int]:
        stats = self._mem(user_id).stats()
        return stats.items, stats.approx_bytes
