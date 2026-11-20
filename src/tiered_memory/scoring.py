"""Decay, reinforcement, retention and TTL arithmetic shared by retrieval and eviction.

* **Recency** halves every ``half_life`` seconds since the item was last accessed (the Generative Agents paper uses
  an exponential decay of 0.995 per hour; a half-life is the same curve with a readable parameter).
* **Strength** grows with every access and every repeated mention, and slows decay: reinforcement on access.
* **Retention** combines importance, decayed strength and confidence; eviction removes the lowest first.
* **Expiry** is a TTL measured from the last access, stretched by importance and strength. Items at or above the
  protection threshold never expire by TTL (they can still be evicted by a hard capacity limit).
"""

from __future__ import annotations

import math


def recency(now: float, last_accessed: float, half_life_seconds: float) -> float:
    """Exponential recency in (0, 1]: 1 when just accessed, 0.5 after one half-life."""
    age = max(0.0, now - last_accessed)
    return float(0.5 ** (age / half_life_seconds))


def strength(access_count: int, mentions: int = 1) -> float:
    """Reinforcement factor >= 1 that grows logarithmically with accesses and repeated mentions."""
    return 1.0 + math.log1p(max(0, access_count) + max(0, mentions - 1))


def retention(
    *,
    importance: float,
    now: float,
    last_accessed: float,
    half_life_seconds: float,
    access_count: int = 0,
    mentions: int = 1,
    confidence: float = 1.0,
) -> float:
    """Score used to pick eviction victims (lowest first): 0.5·importance + 0.4·decay + 0.1·confidence."""
    decay = recency(now, last_accessed, half_life_seconds * strength(access_count, mentions))
    return 0.5 * importance + 0.4 * decay + 0.1 * confidence


def expiry(
    *,
    last_accessed: float,
    ttl_seconds: float | None,
    importance: float,
    access_count: int = 0,
    mentions: int = 1,
    importance_gain: float = 3.0,
    protect_importance: float = 0.9,
) -> float | None:
    """Expiry time: ``last_accessed + ttl × (1 + gain × importance) × strength``; None means never."""
    if ttl_seconds is None or importance >= protect_importance:
        return None
    return last_accessed + ttl_seconds * (1.0 + importance_gain * importance) * strength(access_count, mentions)
