from __future__ import annotations

import pytest

from tiered_memory.extraction import LLMExtractor, RuleBasedExtractor, normalise_date
from tiered_memory.llm import BaseLLM, LLMError, Messages

X = RuleBasedExtractor()


def facts(sentence: str) -> set[tuple[str, str, bool]]:
    return {(c.predicate, c.object, c.negated) for c in X.extract_sentence(sentence)}


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("I live in Turin.", ("lives_in", "Turin")),
        ("I just moved to Milan last weekend.", ("lives_in", "Milan")),
        ("We relocated to Frankfurt am Main this month.", ("lives_in", "Frankfurt am Main")),
        ("I'm based in Lyon these days.", ("lives_in", "Lyon")),
        ("My manager is Paolo Ferri.", ("manager", "Paolo Ferri")),
        ("Giulia Conti is my new manager.", ("manager", "Giulia Conti")),
        ("I now report to Anna Rossi.", ("manager", "Anna Rossi")),
        ("I joined Northwind Labs last month.", ("works_at", "Northwind Labs")),
        ("I work as a backend engineer at Helix Robotics.", ("job_title", "backend engineer")),
        ("My favourite cuisine is Thai.", ("favourite:cuisine", "Thai")),
        ("My favorite cuisine is now Ethiopian food.", ("favourite:cuisine", "Ethiopian food")),
        ("I prefer morning meetings.", ("prefers:meeting", "morning meetings")),
        ("I prefer video meetings now.", ("prefers:meeting", "video meetings")),
        ("I'd prefer afternoon meetings if possible.", ("prefers:meeting", "afternoon meetings")),
        ("I live in Turin, if that matters.", ("lives_in", "Turin")),
        ("My dog is called Biscuit.", ("dog", "Biscuit")),
        ("I'm 34 years old.", ("age", "34")),
        ("My birthday is on 12th of October.", ("birthday", "12 October")),
        ("The quarterly report deadline is 20 March.", ("deadline:quarterly report", "20 March")),
        ("Also the budget review is due on March 3rd.", ("deadline:budget review", "3 March")),
        ("I need to submit the grant proposal by 5 May.", ("deadline:grant proposal", "5 May")),
    ],
)
def test_first_person_patterns(sentence: str, expected: tuple[str, str]) -> None:
    assert (*expected, False) in facts(sentence)


def test_compound_sentence_yields_several_facts() -> None:
    found = facts("I live in Turin and work at Fiat as a data analyst.")
    assert {("lives_in", "Turin", False), ("works_at", "Fiat", False), ("job_title", "data analyst", False)} <= found


def test_lists_are_split_and_clauses_dropped() -> None:
    assert facts("I'm allergic to shellfish and penicillin.") == {
        ("allergic_to", "shellfish", False),
        ("allergic_to", "penicillin", False),
    }
    assert facts("I'm allergic to peanuts, so please keep that in mind.") == {("allergic_to", "peanuts", False)}


@pytest.mark.parametrize(
    "sentence",
    [
        "I might move to Berlin one day.",
        "If I lived in Paris I would cycle everywhere.",
        "My friend Marco lives in Paris.",
        "Do I live in Turin?",
        "I love that idea!",
        "I'd like to book a table.",
        "I'm a big fan of jazz.",
        "My friend Lena's manager is Karl Berg.",
    ],
)
def test_no_facts_from_hypotheticals_questions_or_other_people(sentence: str) -> None:
    assert facts(sentence) == set()


def test_negation_and_correction_flags() -> None:
    assert facts("I don't live in Rome anymore.") == {("lives_in", "Rome", True)}
    assert facts("I left Fiat.") == {("works_at", "Fiat", True)}
    [candidate] = X.extract_sentence("Actually, the quarterly report deadline moved to 27 March.")
    assert candidate.correction and candidate.predicate == "deadline:quarterly report"


def test_task_free_deadline_is_left_for_resolution() -> None:
    assert facts("The deadline has been pushed to 2 June.") == {("deadline:?", "2 June", False)}


def test_span_keeps_surface_form_for_masking() -> None:
    [candidate] = X.extract_sentence("Also the budget review is due on March 3rd.")
    assert candidate.object == "3 March"
    assert candidate.span.lower().startswith("march 3")


@pytest.mark.parametrize(
    ("raw", "normalised"),
    [("March 27th", "27 March"), ("27th of March", "27 March"), ("Sept 5, 2026", "5 September 2026"),
     ("2026-03-27", "2026-03-27"), ("next Friday", "next Friday")],
)
def test_normalise_date(raw: str, normalised: str) -> None:
    assert normalise_date(raw) == normalised


class ScriptedLLM(BaseLLM):
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    def complete(self, messages: Messages, *, json_mode: bool = False, temperature: float = 0.2,
                 max_tokens: int = 1200) -> str:
        self.calls += 1
        if not self.replies:
            raise LLMError("no more scripted replies")
        return self.replies.pop(0)


def test_llm_extractor_maps_predicates_and_dates() -> None:
    llm = ScriptedLLM(['{"facts": [{"sentence_index": 1, "predicate": "Resides in", "object": "Milan"},'
                       ' {"sentence_index": 0, "predicate": "deadline:Q3 report", "object": "March 27th"}]}'])
    out = LLMExtractor(llm).extract([("The Q3 report is due soon.", 1.0), ("We are in Milan now.", 2.0)])
    assert [(c.predicate, c.object) for c in out[1]] == [("lives_in", "Milan")]
    assert [(c.predicate, c.object, c.ts) for c in out[0]] == [("deadline:q3 report", "27 March", 1.0)]


def test_llm_extractor_falls_back_to_rules_when_the_model_fails() -> None:
    out = LLMExtractor(ScriptedLLM([])).extract([("I live in Turin.", 0.0)])
    assert [(c.predicate, c.object) for c in out[0]] == [("lives_in", "Turin")]
