from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from tiered_memory import HashingEmbedder, ManualClock, MemoryConfig, SQLiteStore
from tiered_memory.models import FactCandidate
from tiered_memory.semantic import SemanticMemory, noisy_or

DAY = 86400.0


def semantic(**config: object) -> SemanticMemory:
    return SemanticMemory(SQLiteStore(), HashingEmbedder(), MemoryConfig(**config), "u")  # type: ignore[arg-type]


def cand(predicate: str, obj: str, ts: float, **kw: object) -> FactCandidate:
    return FactCandidate(predicate=predicate, object=obj, ts=ts, sentence=f"{predicate} {obj}", **kw)  # type: ignore[arg-type]


def test_add_then_reinforce_same_fact() -> None:
    sem = semantic()
    first = sem.apply(cand("lives_in", "Turin", 1.0, confidence=0.8), now=1.0, importance=0.4)
    again = sem.apply(cand("lives_in", "turin", 5.0, confidence=0.8), now=5.0, importance=0.6)
    assert (first.action, again.action) == ("added", "reinforced")
    [fact] = sem.current()
    assert fact.mentions == 2
    assert abs(fact.confidence - noisy_or(0.8, 0.8)) < 1e-6
    assert fact.importance == 0.6 and fact.valid_from == 1.0


def test_newer_value_supersedes_and_keeps_history() -> None:
    sem = semantic()
    sem.apply(cand("lives_in", "Turin", 1 * DAY), now=1 * DAY, importance=0.4)
    change = sem.apply(cand("lives_in", "Milan", 9 * DAY), now=9 * DAY, importance=0.4)
    assert change.action == "superseded" and change.replaced_text == "The user lives in Turin."
    [now] = sem.current()
    assert now.object == "Milan" and now.valid_to is None
    old, new = sem.history("user", "lives_in")
    assert (old.object, old.status, old.valid_to, old.superseded_by) == ("Turin", "superseded", 9 * DAY, new.id)
    assert [f.object for f in sem.as_of(5 * DAY)] == ["Turin"]
    assert [f.object for f in sem.as_of(10 * DAY)] == ["Milan"]


def test_late_arriving_older_statement_becomes_history() -> None:
    sem = semantic()
    sem.apply(cand("manager", "Giulia Conti", 10 * DAY), now=10 * DAY, importance=0.4)
    change = sem.apply(cand("manager", "Paolo Ferri", 2 * DAY), now=11 * DAY, importance=0.4)
    assert change.action == "historical"
    assert [f.object for f in sem.current()] == ["Giulia Conti"]
    old = [f for f in sem.history() if f.object == "Paolo Ferri"][0]
    assert old.valid_to == 10 * DAY and old.status == "superseded"


def test_multi_valued_facts_accumulate_but_opposites_conflict() -> None:
    sem = semantic()
    sem.apply(cand("likes", "hiking", 1.0), now=1.0, importance=0.4)
    sem.apply(cand("likes", "jazz", 2.0), now=2.0, importance=0.4)
    assert {f.object for f in sem.current()} == {"hiking", "jazz"}
    change = sem.apply(cand("dislikes", "jazz", 3.0), now=3.0, importance=0.4)
    assert change.action == "superseded"
    assert {(f.predicate, f.object) for f in sem.current()} == {("likes", "hiking"), ("dislikes", "jazz")}


def test_negation_retracts_without_replacement() -> None:
    sem = semantic()
    sem.apply(cand("lives_in", "Rome", 1.0), now=1.0, importance=0.4)
    change = sem.apply(cand("lives_in", "Rome", 2.0, negated=True), now=2.0, importance=0.4)
    assert change.action == "retracted" and sem.current() == []
    assert sem.history()[0].status == "retracted"
    assert sem.apply(cand("lives_in", "Paris", 3.0, negated=True), now=3.0, importance=0.4).action == "ignored"


def test_task_free_deadline_attaches_to_latest_deadline() -> None:
    sem = semantic()
    sem.apply(cand("deadline:atlas migration", "14 April", 1.0), now=1.0, importance=0.5)
    change = sem.apply(cand("deadline:?", "2 June", 5.0), now=5.0, importance=0.5)
    assert change.action == "superseded"
    assert [(f.predicate, f.object) for f in sem.current()] == [("deadline:atlas migration", "2 June")]


def test_low_confidence_candidates_are_ignored() -> None:
    sem = semantic(min_fact_confidence=0.6)
    assert sem.apply(cand("lives_in", "Oslo", 1.0, confidence=0.3), now=1.0, importance=0.5).action == "ignored"
    assert sem.current() == []


def test_supersession_can_be_disabled_for_ablation() -> None:
    sem = semantic(supersede=False)
    sem.apply(cand("lives_in", "Turin", 1.0), now=1.0, importance=0.4)
    sem.apply(cand("lives_in", "Milan", 2.0), now=2.0, importance=0.4)
    assert {f.object for f in sem.current()} == {"Turin", "Milan"}


def test_reapplying_the_same_statement_is_idempotent() -> None:
    sem = semantic()
    sem.apply(cand("lives_in", "Turin", 1.0), now=1.0, importance=0.4)
    sem.apply(cand("lives_in", "Milan", 2.0), now=2.0, importance=0.4)
    sem.apply(cand("lives_in", "Turin", 1.0), now=3.0, importance=0.4)
    assert len(sem.history()) == 2


CITIES = ["Turin", "Milan", "Lyon", "Porto"]


@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(st.sampled_from(CITIES), st.integers(0, 50)), min_size=1, max_size=12))
def test_validity_intervals_never_overlap(updates: list[tuple[str, int]]) -> None:
    """Whatever order statements arrive in, one current value per slot and history intervals do not overlap."""
    sem = semantic()
    for step, (city, day) in enumerate(updates):
        sem.apply(cand("lives_in", city, day * DAY), now=100 * DAY + step, importance=0.4)
    current = sem.current()
    assert len(current) == 1
    latest = max(day for _, day in updates)
    assert current[0].valid_from <= latest * DAY
    spans = [(f.valid_from, f.valid_to if f.valid_to is not None else float("inf")) for f in sem.history()]
    assert all(start <= end for start, end in spans)
    intervals = sorted(span for span in spans if span[1] > span[0])  # empty intervals were never valid
    for (_, end_a), (start_b, _) in zip(intervals, intervals[1:], strict=False):
        assert end_a <= start_b


def test_clock_type_is_independent_of_semantic_memory() -> None:
    clock = ManualClock(0)
    assert clock.advance(days=1) == DAY


def test_shortened_task_names_resolve_to_the_known_deadline() -> None:
    sem = semantic()
    sem.apply(cand("deadline:quarterly report", "20 March", 1.0), now=1.0, importance=0.5)
    change = sem.apply(cand("deadline:report", "27 March", 2.0), now=2.0, importance=0.5)
    assert change.action == "superseded"
    assert [(f.predicate, f.object) for f in sem.current()] == [("deadline:quarterly report", "27 March")]
    sem.apply(cand("deadline:budget report", "1 May", 3.0), now=3.0, importance=0.5)
    ambiguous = sem.apply(cand("deadline:report", "2 May", 4.0), now=4.0, importance=0.5)
    assert ambiguous.action == "added"  # two known tasks contain "report": no guessing
