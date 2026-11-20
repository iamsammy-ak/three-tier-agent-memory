from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from tiered_memory import ManualClock, MemorySystem
from tiered_memory.scoring import expiry, recency, retention, strength

from .conftest import make_memory, say

DAY = 86400.0


def test_recency_halves_every_half_life() -> None:
    assert recency(0, 0, 10) == 1.0
    assert abs(recency(10, 0, 10) - 0.5) < 1e-12
    assert abs(recency(30, 0, 10) - 0.125) < 1e-12
    assert recency(-5, 0, 10) == 1.0  # clock skew never produces a value above 1


def test_strength_and_expiry_grow_with_reinforcement() -> None:
    assert strength(0) == 1.0 and strength(3) > strength(1) > strength(0)
    base = expiry(last_accessed=0, ttl_seconds=DAY, importance=0.2)
    reinforced = expiry(last_accessed=0, ttl_seconds=DAY, importance=0.2, access_count=3)
    assert base is not None and reinforced is not None and reinforced > base
    assert expiry(last_accessed=0, ttl_seconds=DAY, importance=0.95) is None  # protected
    assert expiry(last_accessed=0, ttl_seconds=None, importance=0.1) is None


@settings(max_examples=50)
@given(
    st.floats(0, 1), st.floats(0, 1), st.integers(0, 50), st.floats(0, 1e7), st.floats(0, 1e7)
)
def test_retention_is_bounded_and_monotone_in_importance(imp_a: float, imp_b: float, accesses: int, age: float,
                                                         extra: float) -> None:
    low, high = sorted((imp_a, imp_b))
    common = {"now": age + extra, "last_accessed": extra, "half_life_seconds": DAY, "access_count": accesses}
    r_low, r_high = retention(importance=low, **common), retention(importance=high, **common)
    assert 0.0 <= r_low <= r_high <= 1.0


def test_ttl_expiry_removes_small_talk_but_keeps_protected_memories(clock: ManualClock) -> None:
    memory = make_memory(clock, episode_ttl_days=7)
    say(memory, clock, "s1", "We talked about films for a while.")
    memory.end_session("s1")
    memory.remember("Remember that I'm allergic to peanuts.")
    clock.advance(days=60)
    report = memory.maintain()
    assert len(report.expired) == 1
    [survivor] = memory.episodes()
    assert survivor.reason == "note" and survivor.expires_at is None
    [tombstone] = memory.tombstones()
    assert (tombstone.tier, tombstone.reason) == ("episodic", "ttl")
    assert not hasattr(tombstone, "text")  # tombstones never keep content


def test_capacity_eviction_drops_lowest_retention_first(clock: ManualClock) -> None:
    memory = make_memory(clock, episode_capacity=2, auto_maintain=False)
    texts = ["We talked about films for a while.", "Remember that I'm allergic to peanuts.",
             "I'm presenting at DataSummit on 14 May."]
    for i, text in enumerate(texts):
        clock.advance(hours=1)
        say(memory, clock, f"s{i}", text)
        memory.end_session(f"s{i}")
    report = memory.maintain()
    assert len(report.evicted) == 1
    remaining = " ".join(e.full_text() for e in memory.episodes())
    assert "films" not in remaining and "peanuts" in remaining


def test_superseded_history_expires_after_history_ttl(clock: ManualClock) -> None:
    memory = make_memory(clock, history_ttl_days=30)
    memory.remember("I live in Turin.")
    clock.advance(days=1)
    memory.remember("I just moved to Milan.")
    clock.advance(days=31)
    memory.maintain()
    assert [f.object for f in memory.facts(include_history=True)] == ["Milan"]
    assert any(t.reason == "history_ttl" for t in memory.tombstones())


def test_eviction_can_be_disabled(clock: ManualClock) -> None:
    memory = make_memory(clock, eviction=False, episode_ttl_days=1)
    say(memory, clock, "s1", "We talked about films for a while.")
    memory.end_session("s1")
    clock.advance(days=365)
    assert memory.maintain().removed == 0 and len(memory.episodes()) == 1


def test_forgotten_facts_are_not_relearned(memory: MemorySystem, clock: ManualClock) -> None:
    memory.remember("I live in Turin.")
    [fact] = memory.facts()
    assert memory.forget(fact.id)
    assert memory.facts() == [] and memory.tombstones()[-1].reason == "forgotten"
    clock.advance(days=1)
    report = memory.remember("I live in Turin.")
    assert [c.action for c in report.facts] == ["blocked"]
    assert memory.facts() == []
    assert not memory.forget("f-does-not-exist")


def test_history_of_an_imported_old_transcript_is_kept(clock: ManualClock) -> None:
    memory = make_memory(clock, history_ttl_days=30)
    long_ago = clock.now() - 200 * DAY
    memory.observe("old", "user", "I live in Turin.", ts=long_ago)
    memory.observe("old", "user", "I just moved to Milan.", ts=long_ago + DAY)
    memory.end_session("old")  # consolidated today, about events from 200 days ago
    assert [f.status for f in memory.history("lives_in")] == ["superseded", "current"]
    clock.advance(days=31)
    memory.maintain()
    assert [f.object for f in memory.facts(include_history=True)] == ["Milan"]
