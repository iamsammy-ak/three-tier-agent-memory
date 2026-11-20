"""Pydantic data contracts shared by the store, the tiers, the retrieval API and the CLI."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Tier = Literal["working", "episodic", "semantic"]
Role = Literal["user", "assistant", "system", "goal", "note"]
WorkingKind = Literal["goal", "note", "turn"]
FactStatus = Literal["current", "superseded", "retracted"]
EpisodeReason = Literal["overflow", "session_end", "abandoned", "note"]
FactAction = Literal["added", "reinforced", "merged", "superseded", "historical", "retracted", "blocked", "ignored"]


def fmt_ts(ts: float | None) -> str:
    """Format a UNIX timestamp as a short UTC date-time string (``-`` for None)."""
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def fmt_day(ts: float | None) -> str:
    """Format a UNIX timestamp as a UTC date (``-`` for None)."""
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


class ImportanceScore(BaseModel):
    """Importance in [0, 1] with the reasons that produced it (for explainability)."""

    value: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)


class WorkingItem(BaseModel):
    """One entry of a session scratchpad: a goal, a pinned note or a conversation turn."""

    id: int
    namespace: str
    session_id: str
    kind: WorkingKind
    role: Role
    text: str
    ts: float
    importance: float
    reasons: list[str] = Field(default_factory=list)
    tokens: int


class EpisodeSentence(BaseModel):
    """A sentence kept verbatim inside an episode, with its timestamp and links to facts it produced."""

    text: str
    ts: float
    role: Role
    importance: float
    salient: bool = False
    fact_ids: list[str] = Field(default_factory=list)
    spans: dict[str, str] = Field(default_factory=dict, description="fact id -> object text as written")


class Episode(BaseModel):
    """A compressed conversation segment: the kept sentences plus optional abstractive summary."""

    id: str
    namespace: str
    session_id: str
    reason: EpisodeReason
    start_ts: float
    end_ts: float
    sentences: list[EpisodeSentence]
    abstract: str | None = None
    importance: float
    raw_tokens: int
    tokens: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    expires_at: float | None = None

    @property
    def fact_ids(self) -> list[str]:
        return [fid for s in self.sentences for fid in s.fact_ids]

    def render(self, hidden_fact_ids: set[str] | None = None) -> str:
        """Text of the episode with superseded information removed.

        A sentence whose facts are all in ``hidden_fact_ids`` is dropped; in a sentence that also states a current
        fact, only the stale value is masked ("I live in [superseded] and work at Fiat"). The abstractive summary
        is used only when nothing needs hiding, because an abstract cannot be redacted sentence by sentence.
        """
        hidden = hidden_fact_ids or set()
        touched = [any(f in hidden for f in s.fact_ids) for s in self.sentences]
        if self.abstract and not any(touched):
            return self.abstract
        parts: list[str] = []
        for sentence, is_touched in zip(self.sentences, touched, strict=True):
            if not is_touched:
                parts.append(sentence.text)
                continue
            if all(f in hidden for f in sentence.fact_ids):
                continue
            text = sentence.text
            for fact_id in sentence.fact_ids:
                span = sentence.spans.get(fact_id)
                if fact_id in hidden and span:
                    text = re.sub(re.escape(span), "[superseded]", text, flags=re.I)
            parts.append(text)
        return " ".join(parts)

    def full_text(self) -> str:
        return " ".join(s.text for s in self.sentences)


class Fact(BaseModel):
    """An atomic subject-predicate-object fact with a validity interval [valid_from, valid_to)."""

    model_config = ConfigDict(frozen=True)

    id: str
    namespace: str
    subject: str
    predicate: str
    object: str
    text: str
    confidence: float = Field(ge=0, le=1)
    importance: float = Field(ge=0, le=1)
    mentions: int = 1
    source_episode: str | None = None
    source_session: str | None = None
    source_text: str = ""
    valid_from: float
    valid_to: float | None = None
    status: FactStatus = "current"
    superseded_by: str | None = None
    created_at: float
    last_accessed: float
    access_count: int = 0
    expires_at: float | None = None


class FactCandidate(BaseModel):
    """A fact proposed by an extractor, before deduplication and contradiction handling."""

    subject: str = "user"
    predicate: str
    object: str
    negated: bool = False
    confidence: float = Field(0.8, ge=0, le=1)
    correction: bool = False
    sentence: str = ""
    span: str = Field("", description="The object as written in the sentence (before normalisation).")
    ts: float | None = Field(None, description="When the statement was made; None means 'now'.")


class FactChange(BaseModel):
    """What happened to semantic memory when a candidate was applied."""

    action: FactAction
    fact_id: str | None = None
    text: str
    replaced_id: str | None = None
    replaced_text: str | None = None


class Tombstone(BaseModel):
    """Marker left behind when an item is deleted: keeps no content, only a hash, so deletions stay deleted."""

    id: str
    namespace: str
    tier: Tier
    label: str
    reason: str
    content_hash: str
    created_at: float
    deleted_at: float


class MemoryEvent(BaseModel):
    """Audit-log entry (episode created, fact superseded, item evicted, ...)."""

    ts: float
    kind: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ScoreBreakdown(BaseModel):
    """Components of the retrieval score. Relevance and recency are min-max normalised over the candidates."""

    relevance: float
    relevance_raw: float
    recency: float
    importance: float
    penalty: float = 1.0
    total: float


class Provenance(BaseModel):
    """Where a recalled item came from and how much to trust it."""

    tier: Tier
    session_id: str | None = None
    source_episode: str | None = None
    created_at: float | None = None
    valid_from: float | None = None
    valid_to: float | None = None
    status: str | None = None
    confidence: float | None = None
    mentions: int | None = None
    access_count: int = 0
    note: str | None = None


class RecallItem(BaseModel):
    """One packed item of a recall result."""

    id: str
    tier: Tier
    text: str
    tokens: int
    score: ScoreBreakdown
    provenance: Provenance


class RecallResult(BaseModel):
    """Output of ``MemorySystem.recall``: ranked items that fit the token budget, with provenance and scores."""

    query: str
    budget_tokens: int
    used_tokens: int
    items: list[RecallItem]
    considered: int
    skipped_for_budget: int = 0

    def as_context(self) -> str:
        """Render the packed items as a prompt block grouped by tier."""
        sections: list[str] = []
        titles = {"semantic": "Known facts", "episodic": "Relevant past conversations", "working": "Current session"}
        for tier in ("semantic", "episodic", "working"):
            lines = []
            for item in (i for i in self.items if i.tier == tier):
                prov = item.provenance
                if tier == "semantic":
                    span = f"since {fmt_day(prov.valid_from)}"
                    if prov.valid_to is not None:
                        span = f"{fmt_day(prov.valid_from)} to {fmt_day(prov.valid_to)}, no longer true"
                    lines.append(f"- {item.text} ({span})")
                elif tier == "episodic":
                    lines.append(f"- [{fmt_day(prov.created_at)}] {item.text}")
                else:
                    lines.append(f"- {item.text}")
            if lines:
                sections.append(f"## {titles[tier]}\n" + "\n".join(lines))
        return "\n\n".join(sections)


class ConsolidationReport(BaseModel):
    """Result of turning working-memory items into an episode and facts."""

    session_id: str
    reason: EpisodeReason
    episode_id: str | None = None
    raw_tokens: int = 0
    kept_tokens: int = 0
    sentences_in: int = 0
    sentences_kept: int = 0
    facts: list[FactChange] = Field(default_factory=list)


class MaintenanceReport(BaseModel):
    """Result of a maintenance pass: abandoned sessions consolidated and items expired or evicted."""

    now: float
    consolidated: list[ConsolidationReport] = Field(default_factory=list)
    expired: list[str] = Field(default_factory=list)
    evicted: list[str] = Field(default_factory=list)

    @property
    def removed(self) -> int:
        return len(self.expired) + len(self.evicted)


class ObserveResult(BaseModel):
    """Result of adding one turn to working memory."""

    item_id: int
    importance: ImportanceScore
    working_tokens: int
    consolidation: ConsolidationReport | None = None


class StoreStats(BaseModel):
    """Size of one namespace in the store."""

    namespace: str
    working_items: int
    episodes: int
    facts_current: int
    facts_history: int
    tombstones: int
    open_sessions: int
    approx_bytes: int

    @property
    def items(self) -> int:
        return self.working_items + self.episodes + self.facts_current + self.facts_history
