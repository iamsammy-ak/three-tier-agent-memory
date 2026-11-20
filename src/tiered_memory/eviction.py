"""TTL expiry and capacity eviction with tombstones.

* **TTL**: an episode or current fact expires when ``expires_at`` passes (see ``scoring.expiry``); every access
  pushes it out again. Superseded or retracted facts are kept as history for ``history_ttl_days`` after they stopped
  being true or after the store learned that, whichever is later (so importing an old transcript keeps its history).
* **Capacity**: when a tier holds more items than its limit, the items with the lowest retention score go first.
  History facts are scored at half their retention, so they are evicted before current facts.
* **Tombstones**: every deletion leaves a content-free marker (id, tier, reason, content hash). Episodes are never
  rebuilt from a tombstoned id, and facts deleted with ``reason="forgotten"`` are never re-learned.
"""

from __future__ import annotations

from .config import DAY, MemoryConfig
from .models import Episode, Fact
from .scoring import retention
from .semantic import SemanticMemory
from .store import SQLiteStore, content_hash


class Evictor:
    """Applies TTLs and capacity limits to one namespace."""

    def __init__(self, store: SQLiteStore, config: MemoryConfig, namespace: str, semantic: SemanticMemory) -> None:
        self.store = store
        self.config = config
        self.namespace = namespace
        self.semantic = semantic

    def delete_episode(self, episode: Episode, *, reason: str, now: float) -> None:
        self.store.delete_item(
            self.namespace,
            "episodic",
            episode.id,
            reason=reason,
            label=f"episode:{episode.session_id}",
            chash=content_hash(self.namespace, episode.full_text()),
            created_at=episode.created_at,
            deleted_at=now,
        )
        self.store.log_event(self.namespace, now, "episode_evicted", id=episode.id, reason=reason)

    def delete_fact(self, fact: Fact, *, reason: str, now: float) -> None:
        self.store.delete_item(
            self.namespace,
            "semantic",
            fact.id,
            reason=reason,
            label=f"fact:{fact.predicate}",
            chash=self.semantic.fact_hash(fact.subject, fact.predicate, fact.object),
            created_at=fact.created_at,
            deleted_at=now,
        )
        self.store.log_event(self.namespace, now, "fact_evicted", id=fact.id, text=fact.text, reason=reason)

    def episode_retention(self, episode: Episode, now: float) -> float:
        return retention(
            importance=episode.importance,
            now=now,
            last_accessed=episode.last_accessed,
            half_life_seconds=self.config.episode_half_life_days * DAY,
            access_count=episode.access_count,
        )

    def fact_retention(self, fact: Fact, now: float) -> float:
        score = retention(
            importance=fact.importance,
            now=now,
            last_accessed=fact.last_accessed,
            half_life_seconds=self.config.fact_half_life_days * DAY,
            access_count=fact.access_count,
            mentions=fact.mentions,
            confidence=fact.confidence,
        )
        return score if fact.status == "current" else 0.5 * score

    def run(self, now: float) -> tuple[list[str], list[str]]:
        """Expire by TTL, then evict by capacity. Returns (expired ids, evicted ids)."""
        if not self.config.eviction:
            return [], []
        expired: list[str] = []
        evicted: list[str] = []
        episodes = []
        for episode in self.store.episodes(self.namespace):
            if episode.expires_at is not None and episode.expires_at <= now:
                self.delete_episode(episode, reason="ttl", now=now)
                expired.append(episode.id)
            else:
                episodes.append(episode)
        facts = []
        history_ttl = self.config.history_ttl_days * DAY
        for fact in self.store.facts(self.namespace):
            if fact.status == "current" and fact.expires_at is not None and fact.expires_at <= now:
                self.delete_fact(fact, reason="ttl", now=now)
                expired.append(fact.id)
            elif fact.status != "current" and max(fact.valid_to or 0.0, fact.last_accessed) + history_ttl <= now:
                self.delete_fact(fact, reason="history_ttl", now=now)
                expired.append(fact.id)
            else:
                facts.append(fact)
        overflow = len(episodes) - self.config.episode_capacity
        if overflow > 0:
            for episode in sorted(episodes, key=lambda e: (self.episode_retention(e, now), e.start_ts))[:overflow]:
                self.delete_episode(episode, reason="capacity", now=now)
                evicted.append(episode.id)
        overflow = len(facts) - self.config.fact_capacity
        if overflow > 0:
            for fact in sorted(facts, key=lambda f: (self.fact_retention(f, now), f.valid_from))[:overflow]:
                self.delete_fact(fact, reason="capacity", now=now)
                evicted.append(fact.id)
        return expired, evicted
