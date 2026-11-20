"""Tunable parameters for the three memory tiers.

Every number that shapes behaviour lives here, so an experiment is a `MemoryConfig` diff rather than a code edit.
Durations are given in human units (hours or days) and converted to seconds where they are used.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

HOUR = 3600.0
DAY = 24 * HOUR


class ScoringWeights(BaseModel):
    """Weights of the Generative-Agents-style retrieval score: w_rel·relevance + w_imp·importance + w_rec·recency."""

    model_config = ConfigDict(frozen=True)

    relevance: float = Field(1.0, ge=0)
    importance: float = Field(0.3, ge=0)
    recency: float = Field(0.2, ge=0)


class MemoryConfig(BaseModel):
    """Configuration for working, episodic and semantic memory plus retrieval and eviction."""

    model_config = ConfigDict(frozen=True)

    token_counter: str = Field("approx", description="'approx' (no download) or 'tiktoken[:encoding]' (tokens extra).")

    # --- working memory (per-session scratchpad)
    working_budget_tokens: int = Field(320, ge=32, description="Token budget of the scratchpad before it overflows.")
    working_keep_ratio: float = Field(
        0.4, gt=0, lt=1, description="After an overflow, the oldest turns are consolidated until this share remains."
    )
    working_min_turns: int = Field(2, ge=0, description="Most recent turns that are never flushed on overflow.")
    working_ttl_hours: float = Field(12.0, gt=0, description="Idle time after which an open session is consolidated.")

    # --- episodic memory (compressed conversation segments)
    summary_ratio: float = Field(0.35, gt=0, le=1, description="Target share of tokens the extractive summary keeps.")
    summary_max_tokens: int = Field(160, ge=16)
    salient_threshold: float = Field(
        0.45, ge=0, le=1, description="Sentences at or above this importance are kept verbatim."
    )
    episode_ttl_days: float = Field(30.0, gt=0)
    episode_half_life_days: float = Field(7.0, gt=0)
    episode_capacity: int = Field(200, ge=1)

    # --- semantic memory (versioned facts)
    fact_ttl_days: float | None = Field(180.0, description="Idle TTL of current facts; None disables it.")
    history_ttl_days: float = Field(90.0, gt=0, description="How long superseded facts are kept as history.")
    fact_half_life_days: float = Field(30.0, gt=0)
    fact_capacity: int = Field(500, ge=1)
    min_fact_confidence: float = Field(0.5, ge=0, le=1)
    merge_similarity: float = Field(
        0.92, ge=0, le=1, description="Embedding similarity above which two multi-valued facts are merged."
    )
    supersede: bool = Field(True, description="Close older facts when a newer contradicting fact arrives.")

    # --- importance, decay and retention
    discard_importance: float = Field(
        0.1, ge=0, le=1, description="Segments whose sentences all score below this are dropped (pure small talk)."
    )
    protect_importance: float = Field(0.9, ge=0, le=1, description="Items this important never expire by TTL.")
    ttl_importance_gain: float = Field(3.0, ge=0, description="TTL multiplier: ttl × (1 + gain × importance).")
    eviction: bool = Field(True, description="Enable TTL expiry and capacity eviction.")
    auto_maintain: bool = Field(True, description="Run TTL/capacity maintenance after every session end.")

    # --- retrieval
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    snippet_sentences: int = Field(2, ge=1, description="Best-matching sentences returned per episode.")
    stale_penalty: float = Field(
        0.5, ge=0, le=1, description="Score reduction for snippets with masked outdated values and for history facts."
    )
    dedupe_similarity: float = Field(0.97, ge=0, le=1)
    reinforce_on_recall: bool = True
