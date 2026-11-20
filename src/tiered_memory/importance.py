"""Importance scoring in the spirit of Generative Agents (Park et al., 2023).

The paper asks a language model to rate each memory's "poignancy" from 1 to 10. ``HeuristicImportance`` is the
offline stand-in: transparent rules that reward what an assistant should not forget (explicit "remember" requests,
preferences, dates and numbers, user corrections, personal facts) and penalise small talk. ``LLMImportance`` asks a
model for the 1-10 rating when a key is available and falls back to the heuristic otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from .extraction import DATE
from .llm import LLM, LLMError
from .models import ImportanceScore

_RULES: list[tuple[str, float, re.Pattern[str]]] = [
    (
        "explicit",
        0.55,
        re.compile(
            r"\b(?:remember|don't forget|do not forget|note that|keep in mind|make a note|for the record|important:|"
            r"just so you know|fyi|for your information)",
            re.I,
        ),
    ),
    (
        "preference",
        0.25,
        re.compile(
            r"\b(?:prefer|favou?rite|i (?:really )?(?:like|love|hate|enjoy|dislike)|can't stand|allergic|allergy)\b",
            re.I,
        ),
    ),
    (
        "correction",
        0.25,
        re.compile(r"^\s*(?:actually|correction|sorry,? i meant|no,? wait|oops)\b|\bi meant\b|\bnot .+ but\b", re.I),
    ),
    (
        "personal-fact",
        0.25,
        re.compile(
            r"\b(?:i (?:live|work|moved|relocated|joined|report|left|quit)|we (?:live|moved|relocated)|"
            r"i'?m (?:based|a|an|allergic)|my (?:name|manager|boss|wife|husband|partner|dog|cat|birthday|job|role|"
            r"new|employer|company)|call me)\b",
            re.I,
        ),
    ),
    ("deadline", 0.2, re.compile(r"\b(?:deadline|due (?:on|by|date)|is due|by (?:monday|tuesday|wednesday|"
                                  r"thursday|friday|the end of))\b", re.I)),
    ("date-or-number", 0.15, re.compile(rf"{DATE}|\b\d+(?:[.,]\d+)?\b")),
]
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|i'd|me|my|mine|we|our)\b", re.I)
_NAMED_ENTITY = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+")
_SMALL_TALK = re.compile(
    r"^(?:\s*(?:hi|hello|hey|thanks|thank you|ok(?:ay)?|cool|great|nice|sure|lol|haha|bye|good (?:morning|night)|"
    r"see you|cheers|sounds good|no worries|got it)\b[\s!.,]*)+",
    re.I,
)
_CHATTER = re.compile(r"\b(?:weather|rain|sunny|traffic|coffee break|weekend|match|game last night|tv|movie|film)\b",
                      re.I)


class ImportanceScorer(Protocol):
    name: str

    def score(self, text: str, role: str = "user") -> ImportanceScore: ...

    def score_many(self, items: Sequence[tuple[str, str]]) -> list[ImportanceScore]: ...


class HeuristicImportance:
    """Offline stand-in for LLM importance ratings: additive rule weights, clamped to [0, 1]."""

    name = "heuristic"

    def __init__(self, base: float = 0.15, assistant_factor: float = 0.5) -> None:
        self.base = base
        self.assistant_factor = assistant_factor

    def score(self, text: str, role: str = "user") -> ImportanceScore:
        value = self.base
        reasons: list[str] = []
        for label, weight, pattern in _RULES:
            if pattern.search(text):
                value += weight
                reasons.append(label)
        if "personal-fact" not in reasons and _FIRST_PERSON.search(text) and _NAMED_ENTITY.search(text.strip()):
            value += 0.2
            reasons.append("personal-mention")
        stripped = _SMALL_TALK.sub("", text).strip()
        if len(stripped) < 3 or (_CHATTER.search(text) and not reasons):
            value -= 0.1
            reasons.append("small-talk")
        if role == "assistant":
            value *= self.assistant_factor
            reasons.append("assistant")
        elif role in ("goal", "note"):
            value = max(value, 0.9 if role == "note" else 0.6)
            reasons.append(role)
        return ImportanceScore(value=round(min(1.0, max(0.0, value)), 3), reasons=reasons)

    def score_many(self, items: Sequence[tuple[str, str]]) -> list[ImportanceScore]:
        return [self.score(text, role) for text, role in items]


class _Ratings(BaseModel):
    ratings: list[int] = Field(default_factory=list)


_RATE_PROMPT = """Rate how important each numbered message is for a personal assistant to remember long term,
on a scale of 1 to 10: 1 is mundane small talk (greetings, weather), 10 is critical and lasting (allergies,
explicit requests to remember, deadlines, where the user lives or works, corrections of earlier facts).
Return {"ratings": [one integer per message, in order]}."""


class LLMImportance:
    """Generative-Agents-style 1-10 ratings from an LLM, rescaled to [0, 1]; heuristic fallback on failure."""

    def __init__(self, llm: LLM, fallback: ImportanceScorer | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or HeuristicImportance()
        self.name = f"llm:{llm.name}"

    def score(self, text: str, role: str = "user") -> ImportanceScore:
        return self.score_many([(text, role)])[0]

    def score_many(self, items: Sequence[tuple[str, str]]) -> list[ImportanceScore]:
        if not items:
            return []
        numbered = "\n".join(f"{i}. [{role}] {text}" for i, (text, role) in enumerate(items))
        try:
            reply = self.llm.complete_json(
                [{"role": "system", "content": _RATE_PROMPT}, {"role": "user", "content": numbered}], _Ratings
            )
        except LLMError:
            return self.fallback.score_many(items)
        if len(reply.ratings) != len(items):
            return self.fallback.score_many(items)
        return [
            ImportanceScore(value=round((min(10, max(1, r)) - 1) / 9, 3), reasons=["llm"]) for r in reply.ratings
        ]
