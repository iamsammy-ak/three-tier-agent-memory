"""``MemorySystem``: the public facade over working, episodic and semantic memory."""

from __future__ import annotations

import uuid
from pathlib import Path

from .clock import Clock, SystemClock
from .compression import Compressor, ExtractiveCompressor, LLMCompressor
from .config import HOUR, MemoryConfig
from .consolidation import Consolidator
from .embeddings import Embedder, HashingEmbedder
from .eviction import Evictor
from .extraction import FactExtractor, LLMExtractor, RuleBasedExtractor
from .importance import HeuristicImportance, ImportanceScorer, LLMImportance
from .llm import LLM, get_llm
from .models import (
    ConsolidationReport,
    Episode,
    EpisodeReason,
    Fact,
    MaintenanceReport,
    MemoryEvent,
    ObserveResult,
    RecallResult,
    StoreStats,
    Tombstone,
    WorkingItem,
)
from .retrieval import Retriever
from .semantic import SemanticMemory
from .store import SQLiteStore
from .tokens import TokenCounter, get_token_counter
from .working import WorkingMemory


class MemorySystem:
    """Three-tier memory for one namespace (a user, an agent, or a user-agent pair).

    Components are chosen automatically: with ``OPENROUTER_API_KEY`` set (and ``offline`` False) an LLM writes
    summaries, extracts facts and rates importance; otherwise the offline stand-ins do (TextRank, rules, heuristics).
    Any component can be passed explicitly.
    """

    def __init__(
        self,
        store: SQLiteStore | None = None,
        *,
        namespace: str = "default",
        config: MemoryConfig | None = None,
        embedder: Embedder | None = None,
        clock: Clock | None = None,
        token_counter: TokenCounter | None = None,
        llm: LLM | None = None,
        offline: bool = False,
        importance: ImportanceScorer | None = None,
        compressor: Compressor | None = None,
        extractor: FactExtractor | None = None,
    ) -> None:
        if not namespace.strip():
            raise ValueError("namespace must not be empty")
        self.store = store or SQLiteStore()
        self.namespace = namespace
        self.config = config or MemoryConfig()
        self.embedder = embedder or HashingEmbedder()
        self.clock = clock or SystemClock()
        self.counter = token_counter or get_token_counter(self.config.token_counter)
        self.llm = None if offline else (llm or get_llm())
        extractive = ExtractiveCompressor(
            self.counter,
            ratio=self.config.summary_ratio,
            max_tokens=self.config.summary_max_tokens,
            min_importance=self.config.discard_importance,
        )
        if self.llm is not None:
            self.importance = importance or LLMImportance(self.llm)
            self.compressor = compressor or LLMCompressor(self.llm, extractive)
            self.extractor = extractor or LLMExtractor(self.llm)
        else:
            self.importance = importance or HeuristicImportance()
            self.compressor = compressor or extractive
            self.extractor = extractor or RuleBasedExtractor()
        self.store.bind_embedder(self.embedder.name, self.embedder.dim)

        self.working = WorkingMemory(self.store, self.counter, self.config, namespace)
        self.semantic = SemanticMemory(self.store, self.embedder, self.config, namespace)
        self.consolidator = Consolidator(
            store=self.store,
            embedder=self.embedder,
            counter=self.counter,
            config=self.config,
            namespace=namespace,
            importance=self.importance,
            compressor=self.compressor,
            extractor=self.extractor,
            semantic=self.semantic,
        )
        self.evictor = Evictor(self.store, self.config, namespace, self.semantic)
        self.retriever = Retriever(
            store=self.store,
            embedder=self.embedder,
            counter=self.counter,
            config=self.config,
            namespace=namespace,
            semantic=self.semantic,
            consolidator=self.consolidator,
        )

    @classmethod
    def open(cls, path: str | Path, *, namespace: str = "default", **kwargs: object) -> MemorySystem:
        """Open (or create) a SQLite file and return the memory for ``namespace``."""
        return cls(SQLiteStore(path), namespace=namespace, **kwargs)  # type: ignore[arg-type]

    def for_namespace(self, namespace: str) -> MemorySystem:
        """Another namespace sharing this store, embedder, clock and components."""
        return MemorySystem(
            self.store,
            namespace=namespace,
            config=self.config,
            embedder=self.embedder,
            clock=self.clock,
            token_counter=self.counter,
            llm=self.llm,
            offline=self.llm is None,
            importance=self.importance,
            compressor=self.compressor,
            extractor=self.extractor,
        )

    def close(self) -> None:
        """Close the underlying SQLite connection (shared by every namespace opened from this store)."""
        self.store.close()

    def __enter__(self) -> MemorySystem:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def mode(self) -> str:
        """Which components are active: ``llm`` or ``offline``."""
        return "offline" if self.llm is None else "llm"

    # ---------------------------------------------------------------- sessions
    def start_session(self, session_id: str | None = None, *, goal: str | None = None) -> str:
        """Open a session (optionally with a goal pinned to its scratchpad) and return its id."""
        sid = session_id or f"s-{uuid.uuid4().hex[:8]}"
        now = self.clock.now()
        self.store.touch_session(self.namespace, sid, now)
        if goal:
            self.set_goal(sid, goal)
        return sid

    def set_goal(self, session_id: str, text: str) -> int:
        """Pin a goal to the session scratchpad."""
        score = self.importance.score(text, "goal")
        return self.working.add(session_id, kind="goal", role="goal", text=text, ts=self.clock.now(),
                                importance=score.value, reasons=score.reasons)

    def note(self, session_id: str, text: str) -> int:
        """Pin a note to the session scratchpad (kept until the session ends, then consolidated)."""
        score = self.importance.score(text, "note")
        return self.working.add(session_id, kind="note", role="note", text=text, ts=self.clock.now(),
                                importance=score.value, reasons=score.reasons)

    def observe(self, session_id: str, role: str, text: str, *, ts: float | None = None) -> ObserveResult:
        """Add a conversation turn. If the scratchpad overflows, the oldest turns are consolidated immediately."""
        if role not in ("user", "assistant", "system"):
            raise ValueError("role must be 'user', 'assistant' or 'system'")
        when = self.clock.now() if ts is None else ts
        score = self.importance.score(text, role)
        item_id = self.working.add(session_id, kind="turn", role=role, text=text, ts=when,
                                   importance=score.value, reasons=score.reasons)
        report = None
        flush, remaining = self.working.overflow(session_id)
        if flush:
            now = max(when, self.clock.now())
            report = self.consolidator.consolidate(session_id, flush, reason="overflow", now=now)
            self.store.delete_working([i.id for i in flush])
        return ObserveResult(item_id=item_id, importance=score, working_tokens=remaining, consolidation=report)

    def end_session(self, session_id: str, *, reason: EpisodeReason = "session_end") -> ConsolidationReport:
        """Consolidate everything left in the scratchpad (turns, goals, notes) and close the session."""
        items = self.working.items(session_id)
        last = max((i.ts for i in items), default=self.clock.now())
        now = max(self.clock.now(), last)
        report = self.consolidator.consolidate(session_id, items, reason=reason, now=now)
        self.store.delete_working([i.id for i in items])
        self.store.end_session(self.namespace, session_id, now)
        if self.config.auto_maintain:
            self.evictor.run(now)
        return report

    def remember(self, text: str) -> ConsolidationReport:
        """Store an explicit memory right away: a protected note episode plus any facts it states."""
        now = self.clock.now()
        sid = f"note-{uuid.uuid4().hex[:8]}"
        self.working.add(sid, kind="note", role="note", text=text, ts=now, importance=1.0, reasons=["explicit"])
        return self.end_session(sid, reason="note")

    # ------------------------------------------------------------------ recall
    def recall(
        self,
        query: str,
        *,
        budget_tokens: int = 256,
        k: int | None = None,
        include_history: bool = False,
        session_id: str | None = None,
        reinforce: bool | None = None,
    ) -> RecallResult:
        """Ranked, token-budgeted memories for ``query`` across all three tiers, with scores and provenance.

        Superseded facts (and episode sentences that only stated them) are excluded unless ``include_history``.
        Returned episodes and facts are reinforced unless ``reinforce`` is False.
        """
        if budget_tokens <= 0:
            raise ValueError("budget_tokens must be positive")
        return self.retriever.recall(
            query,
            now=self.clock.now(),
            budget_tokens=budget_tokens,
            k=k,
            include_history=include_history,
            session_id=session_id,
            reinforce=self.config.reinforce_on_recall if reinforce is None else reinforce,
        )

    # ------------------------------------------------------------- maintenance
    def maintain(self, now: float | None = None) -> MaintenanceReport:
        """Consolidate idle sessions, then apply TTL expiry and capacity eviction."""
        when = self.clock.now() if now is None else now
        report = MaintenanceReport(now=when)
        idle = self.config.working_ttl_hours * HOUR
        for session_id, last_activity in self.store.open_sessions(self.namespace):
            if last_activity + idle <= when:
                items = self.working.items(session_id)
                report.consolidated.append(
                    self.consolidator.consolidate(session_id, items, reason="abandoned", now=when)
                )
                self.store.delete_working([i.id for i in items])
                self.store.end_session(self.namespace, session_id, when)
        report.expired, report.evicted = self.evictor.run(when)
        return report

    def forget(self, item_id: str) -> bool:
        """Delete a fact or episode on request. A forgotten fact is blocked from being learned again."""
        now = self.clock.now()
        fact = self.store.get_fact(item_id)
        if fact is not None and fact.namespace == self.namespace:
            self.evictor.delete_fact(fact, reason="forgotten", now=now)
            return True
        episode = self.store.get_episode(item_id)
        if episode is not None and episode.namespace == self.namespace:
            self.evictor.delete_episode(episode, reason="forgotten", now=now)
            return True
        return False

    # ----------------------------------------------------------------- inspect
    def facts(self, *, include_history: bool = False) -> list[Fact]:
        statuses = None if include_history else ["current"]
        return self.store.facts(self.namespace, statuses=statuses)

    def history(self, predicate: str | None = None, subject: str = "user") -> list[Fact]:
        return self.semantic.history(subject, predicate)

    def episodes(self) -> list[Episode]:
        return self.store.episodes(self.namespace)

    def working_items(self, session_id: str | None = None) -> list[WorkingItem]:
        return self.store.working_items(self.namespace, session_id)

    def scratchpad(self, session_id: str) -> str:
        return self.working.render(session_id)

    def tombstones(self) -> list[Tombstone]:
        return self.store.tombstones(self.namespace)

    def events(self, limit: int = 50) -> list[MemoryEvent]:
        return self.store.events(self.namespace, limit)

    def stats(self) -> StoreStats:
        return self.store.stats(self.namespace)
