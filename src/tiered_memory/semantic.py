"""Semantic memory: atomic facts with deduplication, merging and time-aware contradiction handling.

Each fact carries a validity interval ``[valid_from, valid_to)``. When a newer statement contradicts a current
fact (same single-valued slot, different value), the old fact is *closed* rather than deleted: its ``valid_to`` is
set to the new statement's time, its status becomes ``superseded`` and it points at its replacement. History stays
queryable ("where did the user live in March?") while recall only returns what is true now.
"""

from __future__ import annotations

import numpy as np

from .config import DAY, MemoryConfig
from .embeddings import Embedder
from .models import Fact, FactCandidate, FactChange
from .predicates import OPPOSITES, is_single_valued, normalise_predicate, normalise_text, render_fact
from .scoring import expiry
from .store import SQLiteStore, content_hash

_SELF = {"user", "i", "me", "myself", "the user"}


def noisy_or(a: float, b: float) -> float:
    """Combine two independent confidences: 1 - (1-a)(1-b)."""
    return 1.0 - (1.0 - a) * (1.0 - b)


class SemanticMemory:
    """Fact store for one namespace."""

    def __init__(self, store: SQLiteStore, embedder: Embedder, config: MemoryConfig, namespace: str) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config
        self.namespace = namespace

    # ----------------------------------------------------------------- helpers
    def _fact_id(self, subject: str, predicate: str, obj: str, valid_from: float) -> str:
        digest = content_hash(self.namespace, subject, predicate, normalise_text(obj), f"{valid_from:.3f}")
        return "f-" + digest[:12]

    def fact_hash(self, subject: str, predicate: str, obj: str) -> str:
        """Content hash of a fact regardless of time; tombstones of forgotten facts use it."""
        return content_hash(self.namespace, subject, predicate, normalise_text(obj))

    def fact_expiry(self, last_accessed: float, importance: float, access_count: int, mentions: int) -> float | None:
        ttl = self.config.fact_ttl_days * DAY if self.config.fact_ttl_days is not None else None
        return expiry(
            last_accessed=last_accessed,
            ttl_seconds=ttl,
            importance=importance,
            access_count=access_count,
            mentions=mentions,
            importance_gain=self.config.ttl_importance_gain,
            protect_importance=self.config.protect_importance,
        )

    def _vector(self, fact_id: str) -> np.ndarray | None:
        ids, matrix = self.store.fact_vectors(self.namespace)
        try:
            return matrix[ids.index(fact_id)]
        except ValueError:
            return None

    def _resolve_deadline(self, ts: float) -> str:
        """'The deadline moved to X' without a task: attach it to the most recently stated current deadline."""
        deadlines = [
            f
            for f in self.store.facts(self.namespace, statuses=["current"], subject="user")
            if f.predicate.startswith("deadline:") and f.valid_from <= ts
        ]
        if not deadlines:
            return "deadline:unspecified"
        return max(deadlines, key=lambda f: (f.valid_from, f.last_accessed)).predicate

    def _resolve_task(self, predicate: str, ts: float) -> str:
        """Match a shortened task name to a known one: 'report' -> 'quarterly report' when that is unambiguous."""
        task = predicate.partition(":")[2]
        words = set(task.split())
        known = {
            f.predicate
            for f in self.store.facts(self.namespace, statuses=["current"], subject="user")
            if f.predicate.startswith("deadline:") and f.valid_from <= ts
        }
        if predicate in known or not words:
            return predicate
        matches = [p for p in known if words <= set(p.partition(":")[2].split())]
        return matches[0] if len(matches) == 1 else predicate

    def _competitors(self, subject: str, predicate: str, obj: str) -> list[Fact]:
        """Current facts that share the slot of (subject, predicate, obj)."""
        if is_single_valued(predicate):
            return self.store.facts(self.namespace, statuses=["current"], subject=subject, predicate=predicate)
        same = self.store.facts(self.namespace, statuses=["current"], subject=subject, predicate=predicate)
        opposite = OPPOSITES.get(predicate)
        if opposite:
            target = normalise_text(obj)
            same += [
                f
                for f in self.store.facts(self.namespace, statuses=["current"], subject=subject, predicate=opposite)
                if normalise_text(f.object) == target
            ]
        return same

    # ------------------------------------------------------------------- apply
    def apply(
        self,
        candidate: FactCandidate,
        *,
        now: float,
        importance: float,
        source_episode: str | None = None,
        source_session: str | None = None,
    ) -> FactChange:
        """Merge one extracted candidate into semantic memory and report what changed."""
        subject = "user" if candidate.subject.strip().casefold() in _SELF else candidate.subject.strip()
        predicate = normalise_predicate(candidate.predicate) if candidate.predicate != "deadline:?" else "deadline:?"
        obj = candidate.object.strip()
        ts = now if candidate.ts is None else candidate.ts
        if predicate == "deadline:?":
            predicate = self._resolve_deadline(ts)
        elif predicate.startswith("deadline:") and subject == "user":
            predicate = self._resolve_task(predicate, ts)
        text = render_fact(subject, predicate, obj)
        if candidate.confidence < self.config.min_fact_confidence or not obj:
            return FactChange(action="ignored", text=text)
        if self.store.is_tombstoned(self.namespace, chash=self.fact_hash(subject, predicate, obj), reason="forgotten"):
            return FactChange(action="blocked", text=text)

        target = normalise_text(obj)
        if candidate.negated:
            same = [
                f
                for f in self.store.facts(self.namespace, statuses=["current"], subject=subject, predicate=predicate)
                if normalise_text(f.object) == target
            ]
            for fact in same:
                self.store.update_fact(fact.id, status="retracted", valid_to=ts,
                                       last_accessed=max(fact.last_accessed, now))
                self.store.log_event(self.namespace, now, "fact_retracted", id=fact.id, text=fact.text)
            if same:
                return FactChange(action="retracted", fact_id=same[0].id, text=same[0].text)
            return FactChange(action="ignored", text=text)

        fact_id = self._fact_id(subject, predicate, obj, ts)
        if self.store.fact_exists(fact_id):  # the same statement consolidated twice
            existing = self.store.get_fact(fact_id)
            assert existing is not None
            return FactChange(action="reinforced", fact_id=fact_id, text=existing.text)
        fact = Fact(
            id=fact_id,
            namespace=self.namespace,
            subject=subject,
            predicate=predicate,
            object=obj,
            text=text,
            confidence=candidate.confidence,
            importance=importance,
            source_episode=source_episode,
            source_session=source_session,
            source_text=candidate.sentence,
            valid_from=ts,
            created_at=now,
            last_accessed=max(ts, now),
            expires_at=self.fact_expiry(max(ts, now), importance, 0, 1),
        )
        if is_single_valued(predicate) and self.config.supersede:
            return self._apply_single(fact, candidate, now=now, importance=importance)

        competitors = self._competitors(subject, predicate, obj)
        same = [f for f in competitors if f.predicate == predicate and normalise_text(f.object) == target]
        if same:
            return self._reinforce(same[0], candidate, now=now, importance=importance, action="reinforced")
        vector = self.embedder.embed([text])[0]
        for other_fact in competitors:
            if other_fact.predicate != predicate:
                continue
            other = self._vector(other_fact.id)
            if other is not None and float(other @ vector) >= self.config.merge_similarity:
                return self._reinforce(other_fact, candidate, now=now, importance=importance, action="merged")
        # Multi-valued: only the opposite polarity on the same object conflicts (likes X vs dislikes X).
        conflicting = [f for f in competitors if f.predicate != predicate] if self.config.supersede else []
        self.store.insert_fact(fact, vector)
        if not conflicting:
            self.store.log_event(self.namespace, now, "fact_added", id=fact.id, text=text)
            return FactChange(action="added", fact_id=fact.id, text=text)
        for old in conflicting:
            self._close(old, at=fact.valid_from, by=fact, now=now)
        return FactChange(action="superseded", fact_id=fact.id, text=text, replaced_id=conflicting[0].id,
                          replaced_text=conflicting[0].text)

    def _close(self, old: Fact, *, at: float, by: Fact, now: float) -> None:
        # last_accessed marks when the store learned the fact stopped being true (history TTL counts from it)
        self.store.update_fact(old.id, status="superseded", valid_to=at, superseded_by=by.id,
                               last_accessed=max(old.last_accessed, now))
        self.store.log_event(self.namespace, now, "fact_superseded", id=old.id, text=old.text, by=by.id,
                             new_text=by.text)

    def _apply_single(self, fact: Fact, candidate: FactCandidate, *, now: float, importance: float) -> FactChange:
        """Insert a version into a single-valued slot so validity intervals stay ordered and non-overlapping.

        ``covering`` is the version true at the statement's time and ``following`` the next later version. A
        restatement of the covering value reinforces it; a value that is only known from a later time is extended
        backwards; anything else splits the timeline: the covering version is closed and the new one runs until the
        following version starts (or stays current when there is none).
        """
        ts = fact.valid_from
        versions = [
            f
            for f in self.store.facts(self.namespace, subject=fact.subject, predicate=fact.predicate)
            if f.status in ("current", "superseded") and (f.valid_to is None or f.valid_to > f.valid_from)
        ]  # empty intervals (superseded at the instant they were stated) were never valid
        covering = max(
            (f for f in versions if f.valid_from <= ts and (f.valid_to is None or f.valid_to > ts)),
            key=lambda f: f.valid_from,
            default=None,
        )
        following = min((f for f in versions if f.valid_from > ts), key=lambda f: f.valid_from, default=None)
        target = normalise_text(fact.object)
        if covering is not None and normalise_text(covering.object) == target:
            return self._reinforce(covering, candidate, now=now, importance=importance, action="reinforced")
        if following is not None and normalise_text(following.object) == target:
            if covering is not None:
                self._close(covering, at=ts, by=following, now=now)
            return self._reinforce(following, candidate, now=now, importance=importance, action="reinforced")
        if following is not None:
            fact = fact.model_copy(
                update={"status": "superseded", "valid_to": following.valid_from, "superseded_by": following.id}
            )
        self.store.insert_fact(fact, self.embedder.embed([fact.text])[0])
        if covering is not None:
            self._close(covering, at=ts, by=fact, now=now)
        if following is not None:
            self.store.log_event(self.namespace, now, "fact_historical", id=fact.id, text=fact.text,
                                 current=following.id)
            return FactChange(action="historical", fact_id=fact.id, text=fact.text, replaced_id=following.id,
                              replaced_text=following.text)
        if covering is not None:
            return FactChange(action="superseded", fact_id=fact.id, text=fact.text, replaced_id=covering.id,
                              replaced_text=covering.text)
        self.store.log_event(self.namespace, now, "fact_added", id=fact.id, text=fact.text)
        return FactChange(action="added", fact_id=fact.id, text=fact.text)

    def _reinforce(
        self, fact: Fact, candidate: FactCandidate, *, now: float, importance: float, action: str
    ) -> FactChange:
        mentions = fact.mentions + 1
        new_importance = max(fact.importance, importance)
        last = max(fact.last_accessed, now)
        self.store.update_fact(
            fact.id,
            mentions=mentions,
            confidence=round(noisy_or(fact.confidence, candidate.confidence), 4),
            importance=new_importance,
            last_accessed=last,
            valid_from=fact.valid_from if candidate.ts is None else min(fact.valid_from, candidate.ts),
            expires_at=self.fact_expiry(last, new_importance, fact.access_count, mentions),
        )
        self.store.log_event(self.namespace, now, f"fact_{action}", id=fact.id, text=fact.text, mentions=mentions)
        return FactChange(action="reinforced" if action == "reinforced" else "merged", fact_id=fact.id, text=fact.text)

    # ----------------------------------------------------------------- queries
    def current(self) -> list[Fact]:
        return self.store.facts(self.namespace, statuses=["current"])

    def history(self, subject: str = "user", predicate: str | None = None) -> list[Fact]:
        """Every version of a slot (or of all slots), oldest first."""
        pred = normalise_predicate(predicate) if predicate else None
        return self.store.facts(self.namespace, subject=subject, predicate=pred)

    def as_of(self, ts: float, subject: str = "user") -> list[Fact]:
        """Facts that were valid at time ``ts`` (time travel over the validity intervals)."""
        return [
            f
            for f in self.store.facts(self.namespace, subject=subject, statuses=["current", "superseded"])
            if f.valid_from <= ts and (f.valid_to is None or ts < f.valid_to)
        ]
