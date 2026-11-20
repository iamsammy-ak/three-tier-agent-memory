from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tiered_memory import HashingEmbedder, ManualClock, MemorySystem, SQLiteStore
from tiered_memory.store import EmbedderMismatchError

from .conftest import make_memory, say


def test_overflow_consolidates_oldest_turns(clock: ManualClock) -> None:
    memory = make_memory(clock, working_budget_tokens=40)
    sid = memory.start_session("s1")
    reports = []
    for line in [
        "Hi! I'm Sara and I live in Turin, which is a lovely city in the north of Italy.",
        "The weather is lovely today and I am going for a long walk along the river.",
        "My manager is Paolo Ferri and he runs the analytics team on the third floor.",
        "Remember that I'm allergic to peanuts.",
        "I prefer morning meetings.",
    ]:
        clock.advance(minutes=1)
        result = memory.observe(sid, "user", line)
        if result.consolidation:
            reports.append(result.consolidation)
        assert result.working_tokens <= 40 or result.consolidation is None
    assert reports, "the 40-token scratchpad must overflow"
    assert reports[0].reason == "overflow" and reports[0].episode_id
    remaining = memory.working_items("s1")
    assert len(remaining) >= memory.config.working_min_turns
    assert sum(i.tokens for i in remaining) <= 40


def test_goals_and_notes_are_pinned_and_consolidated_at_session_end(memory: MemorySystem, clock: ManualClock) -> None:
    sid = memory.start_session("s1", goal="Plan the Lisbon offsite")
    memory.note(sid, "Budget is 4000 euros")
    say(memory, clock, sid, "Let's start with flights.")
    pad = memory.scratchpad(sid)
    assert pad.splitlines()[0] == "Goal: Plan the Lisbon offsite" and "Note: Budget is 4000 euros" in pad
    report = memory.end_session(sid)
    assert report.reason == "session_end" and memory.working_items(sid) == []
    [episode] = memory.episodes()
    roles = {s.role for s in episode.sentences}
    assert {"goal", "note"} <= roles
    note = next(s for s in episode.sentences if s.role == "note")
    assert note.importance >= 0.9


def test_end_session_extracts_facts_and_links_sentences(memory: MemorySystem, clock: ManualClock) -> None:
    say(memory, clock, "s1", "I live in Turin and work at Fiat.", "The weather is grey.")
    report = memory.end_session("s1")
    assert {c.action for c in report.facts} == {"added"}
    assert {f.text for f in memory.facts()} == {"The user lives in Turin.", "The user works at Fiat."}
    [episode] = memory.episodes()
    linked = [s for s in episode.sentences if s.fact_ids]
    assert linked[0].text == "I live in Turin and work at Fiat."
    assert set(linked[0].spans.values()) == {"Turin", "Fiat"}
    assert all(f.source_episode == episode.id for f in memory.facts())


def test_consolidating_the_same_segment_twice_is_a_no_op(memory: MemorySystem, clock: ManualClock) -> None:
    say(memory, clock, "s1", "I live in Turin.")
    items = memory.working_items("s1")
    first = memory.consolidator.consolidate("s1", items, reason="session_end", now=clock.now())
    second = memory.consolidator.consolidate("s1", items, reason="session_end", now=clock.now())
    assert first.episode_id and second.episode_id is None
    assert len(memory.episodes()) == 1


def test_pure_small_talk_is_not_stored(memory: MemorySystem, clock: ManualClock) -> None:
    say(memory, clock, "s1", "Ok, thanks!", "Haha.")
    report = memory.end_session("s1")
    assert report.episode_id is None and memory.episodes() == []
    assert memory.events()[-1].kind == "segment_discarded"


def test_remember_stores_a_protected_note_and_facts(memory: MemorySystem) -> None:
    report = memory.remember("I'm allergic to peanuts.")
    assert report.reason == "note"
    [fact] = memory.facts()
    assert fact.predicate == "allergic_to"
    [episode] = memory.episodes()
    assert episode.expires_at is None  # explicit memories are protected from TTL expiry


def test_supersession_across_sessions(memory: MemorySystem, clock: ManualClock) -> None:
    say(memory, clock, "s1", "I live in Turin.", "My manager is Paolo Ferri.")
    memory.end_session("s1")
    clock.advance(days=9)
    say(memory, clock, "s2", "I just moved to Milan last weekend.")
    report = memory.end_session("s2")
    assert [c.action for c in report.facts] == ["superseded"]
    assert {f.object for f in memory.facts()} == {"Milan", "Paolo Ferri"}
    turin, milan = memory.history("lives_in")
    assert turin.valid_to == milan.valid_from


def test_persistence_across_restarts(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    clock = ManualClock(1_000_000)
    first = MemorySystem.open(db, namespace="sara", embedder=HashingEmbedder(), clock=clock, offline=True)
    say(first, clock, "s1", "I live in Turin.")
    first.store.close()
    reopened = MemorySystem.open(db, namespace="sara", embedder=HashingEmbedder(), clock=clock, offline=True)
    assert [w.text for w in reopened.working_items("s1")] == ["I live in Turin."]  # unfinished session survived
    reopened.end_session("s1")
    reopened.store.close()
    third = MemorySystem.open(db, namespace="sara", embedder=HashingEmbedder(), clock=clock, offline=True)
    assert [f.object for f in third.facts()] == ["Turin"]
    assert third.recall("Where does the user live?", reinforce=False).items[0].text == "The user lives in Turin."


def test_namespaces_are_isolated(clock: ManualClock) -> None:
    store = SQLiteStore()
    sara = MemorySystem(store, namespace="sara", embedder=HashingEmbedder(), clock=clock, offline=True)
    marco = sara.for_namespace("marco")
    sara.remember("I live in Turin.")
    marco.remember("I live in Paris.")
    assert [f.object for f in sara.facts()] == ["Turin"]
    assert [f.object for f in marco.facts()] == ["Paris"]
    assert all("Turin" not in i.text for i in marco.recall("Where do I live?").items)
    assert set(store.namespaces()) >= {"sara", "marco"}


def test_store_refuses_a_different_embedder() -> None:
    store = SQLiteStore()
    MemorySystem(store, embedder=HashingEmbedder(384), offline=True)
    with pytest.raises(EmbedderMismatchError):
        MemorySystem(store, embedder=HashingEmbedder(256), offline=True)


def test_invalid_arguments_are_rejected(memory: MemorySystem) -> None:
    with pytest.raises(ValueError):
        memory.observe("s1", "robot", "hello")
    with pytest.raises(ValueError):
        memory.recall("anything", budget_tokens=0)
    with pytest.raises(ValueError):
        MemorySystem(namespace=" ", offline=True)


def test_idle_sessions_are_consolidated_by_maintenance(memory: MemorySystem, clock: ManualClock) -> None:
    say(memory, clock, "s1", "I live in Turin.")
    clock.advance(hours=13)
    report = memory.maintain()
    assert [c.reason for c in report.consolidated] == ["abandoned"]
    assert memory.store.open_sessions("sara") == [] and [f.object for f in memory.facts()] == ["Turin"]


def test_context_manager_closes_the_store(tmp_path: Path) -> None:
    with MemorySystem.open(tmp_path / "m.db", embedder=HashingEmbedder(), offline=True) as memory:
        memory.remember("I live in Turin.")
    with pytest.raises(sqlite3.ProgrammingError):
        memory.store.get_meta("schema_version")
