from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from tiered_memory import HashingEmbedder, ManualClock, MemorySystem
from tiered_memory.retrieval import outdated_value_pattern

from .conftest import START, say


def build(clock: ManualClock) -> MemorySystem:
    memory = MemorySystem(namespace="sara", clock=clock, embedder=HashingEmbedder(), offline=True)
    say(memory, clock, "s1", "I live in Turin and work at Fiat.", "I met Marco at the Turin office on Friday.",
        "I'm presenting at DataSummit on 14 May.", "Remember that I'm allergic to peanuts.")
    memory.end_session("s1")
    clock.advance(days=9)
    say(memory, clock, "s2", "I just moved to Milan last weekend.")
    memory.end_session("s2")
    return memory


def test_current_fact_ranks_first_and_stale_values_are_hidden(clock: ManualClock) -> None:
    memory = build(clock)
    result = memory.recall("Which city does the user live in?", budget_tokens=500, reinforce=False)
    assert result.items[0].text == "The user lives in Milan."
    joined = " ".join(i.text for i in result.items)
    assert "Turin" not in joined  # neither the superseded fact nor the chatter that mentions the old city
    assert "[superseded]" in joined
    masked = next(i for i in result.items if "[superseded]" in i.text)
    assert masked.tier == "episodic" and masked.score.penalty < 1.0


def test_history_mode_returns_superseded_facts_with_intervals(clock: ManualClock) -> None:
    memory = build(clock)
    result = memory.recall("Which city does the user live in?", budget_tokens=500, include_history=True,
                           reinforce=False)
    old = next(i for i in result.items if i.text == "The user lives in Turin.")
    assert old.provenance.status == "superseded" and old.provenance.valid_to is not None
    assert "no longer true" in result.as_context()


def test_episodic_snippets_cover_what_the_extractor_missed(clock: ManualClock) -> None:
    memory = build(clock)
    result = memory.recall("Which conference is the user presenting at?", k=3, reinforce=False)
    hit = next(i for i in result.items if "DataSummit" in i.text)
    assert hit.tier == "episodic" and hit.provenance.source_episode
    assert len(hit.text.split(". ")) <= memory.config.snippet_sentences


def test_working_memory_is_recalled_for_the_open_session(memory: MemorySystem, clock: ManualClock) -> None:
    say(memory, clock, "live", "Let's plan the Lisbon offsite for June.")
    result = memory.recall("offsite plans", session_id="live")
    assert result.items[0].tier == "working" and "Lisbon" in result.items[0].text


def test_k_limit_and_provenance(clock: ManualClock) -> None:
    memory = build(clock)
    result = memory.recall("Tell me about the user", k=2, reinforce=False)
    assert len(result.items) == 2 and result.considered >= 2
    for item in result.items:
        assert item.provenance.tier == item.tier
        assert 0.0 <= item.score.relevance <= 1.0 and 0.0 <= item.score.recency <= 1.0


def test_recall_reinforces_returned_items(clock: ManualClock) -> None:
    memory = build(clock)
    before = {f.id: f for f in memory.facts()}
    clock.advance(days=3)
    result = memory.recall("Which city does the user live in?", k=1)
    fact_id = result.items[0].id
    after = memory.store.get_fact(fact_id)
    assert after is not None and after.access_count == before[fact_id].access_count + 1
    assert after.last_accessed == clock.now()
    assert after.expires_at is not None and before[fact_id].expires_at is not None
    assert after.expires_at > before[fact_id].expires_at


def test_empty_memory_returns_nothing(memory: MemorySystem) -> None:
    result = memory.recall("anything at all")
    assert result.items == [] and result.used_tokens == 0 and result.as_context() == ""


def test_outdated_value_pattern_ignores_values_that_are_current_again() -> None:
    clock = ManualClock(START)
    memory = MemorySystem(namespace="x", clock=clock, embedder=HashingEmbedder(), offline=True)
    for city in ("Turin", "Milan", "Turin"):
        clock.advance(days=1)
        memory.remember(f"I live in {city}.")
    pattern = outdated_value_pattern(memory.facts(include_history=True))
    assert pattern is not None and pattern.search("Milan") and not pattern.search("Turin")


@settings(max_examples=30, deadline=None)
@given(st.integers(min_value=1, max_value=120))
def test_packed_items_never_exceed_the_budget(budget: int) -> None:
    clock = ManualClock(START)
    memory = build(clock)
    result = memory.recall("What do we know about the user?", budget_tokens=budget, reinforce=False)
    assert result.used_tokens == sum(i.tokens for i in result.items) <= budget
    texts = [i.text for i in result.items]
    assert len(texts) == len(set(texts))
