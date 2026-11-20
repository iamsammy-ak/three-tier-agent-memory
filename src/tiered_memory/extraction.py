"""Fact extraction: episodic sentences -> subject-predicate-object candidates.

``RuleBasedExtractor`` is the offline stand-in: hand-written patterns for first-person statements such as
"I live in X", "my manager is Y", "I prefer Z" and deadlines. It deliberately skips questions, hypotheticals
("I might move to Berlin") and third-person statements ("my friend lives in Paris").
``LLMExtractor`` asks a model for the same structure and falls back to the rules if the call fails.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from .llm import LLM, LLMError
from .models import FactCandidate
from .predicates import head_noun, normalise_predicate

Sentence = tuple[str, float]  # (text, timestamp)

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
)
_MONTH_FULL = {
    m[:3].lower(): m
    for m in "January February March April May June July August September October November December".split()
}
DATE = (
    rf"(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})\b(?:\s+\d{{4}})?"
    rf"|(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}})?"
    r"|\d{4}-\d{2}-\d{2}"
    r"|(?:next\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))"
)
PROPER = r"[A-Z][\w'’&.-]*(?:\s+(?:(?:de|del|di|da|am|upon|sur|of)\s+)?[A-Z][\w'’&.-]*)*"
PERSON = r"[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)?"
PHRASE = r"[^.;!?]+"
TASK = r"[A-Za-z0-9][\w-]*(?:\s+[A-Za-z0-9][\w-]*){0,4}?"
JOB_NOUNS = (
    "engineer|analyst|manager|designer|developer|scientist|researcher|consultant|teacher|nurse|doctor|lawyer|"
    "accountant|architect|student|writer|editor|director|officer|specialist|administrator|technician|lead|"
    "programmer|economist|pharmacist|translator|recruiter|marketer|intern"
)
CHANGE = r"(?i:is|has\s+(?:been\s+)?(?:moved|pushed|changed|shifted|postponed)\s+to|(?:was|got)\s+(?:moved|pushed|changed|shifted|postponed)\s+to|moved\s+to|is\s+now|changed\s+to)"  # noqa: E501

_QUESTION = re.compile(r"\?\s*$")
_HYPOTHETICAL = re.compile(
    r"\b(?:might|maybe|perhaps|could|wish|someday|one day|thinking (?:of|about)|planning to|hoping to|"
    r"considering|wondering|dream of|would(?! (?:rather|prefer)))\b"
    r"|^\s*if\b|\bif\s+(?:i|we|my|you|they|he|she)\b",  # "if I lived in Paris" but not "if possible"
    re.I,
)
_CORRECTION = re.compile(r"^\s*(?:actually|correction|sorry,? i meant|no,? wait|to correct|oops)\b|\bi meant\b", re.I)
_LIST_SPLIT = re.compile(r",\s*(?:and\s+)?|\s+and\s+|\s*&\s*")
_CLAUSE_END = re.compile(
    r",?\s+(?:so|but|because|since|though|although|which|when|while|as|and\s+(?:i|we|it|that|they|he|she))\b"
    r"|,\s*(?:please|thanks|just)\b",
    re.I,
)
_TRAILING = re.compile(
    r"\s+(?:now|these days|anymore|any more|too|as well|a lot|so much|very much|at the moment|lately|recently|"
    r"right now|nowadays|in general|generally)\s*$",
    re.I,
)
_PRONOUN_OBJECT = re.compile(r"^(?:it|that|this|these|those|them|you|him|her|how|what|when|where|the way|to)\b", re.I)
_TASK_FILLER = {"also", "so", "oh", "and", "but", "update", "btw", "ok", "okay", "actually", "quick", "now", "just"}
_DETERMINERS = {"the", "my", "our", "a", "an", "this", "that", "your"}


def normalise_date(value: str) -> str:
    """'March 27th' / '27th of March' -> '27 March'; ISO dates and weekdays are kept as written."""
    value = value.strip()
    m = re.fullmatch(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTHS})(?:\s+(\d{{4}}))?", value, re.I)
    if not m:
        m2 = re.fullmatch(rf"({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?", value, re.I)
        if not m2:
            return value
        month, day, year = m2.group(1), m2.group(2), m2.group(3)
    else:
        day, month, year = m.group(1), m.group(2), m.group(3)
    full = _MONTH_FULL[month[:3].lower()]
    return f"{int(day)} {full}" + (f" {year}" if year else "")


def _clean_object(value: str) -> str:
    value = value.strip().strip("\"'“”").rstrip(".,;:!?")
    for _ in range(2):
        value = _TRAILING.sub("", value).strip()
    return re.sub(r"^(?:the)\s+", "", value, flags=re.I)


def _clean_task(value: str) -> str:
    words = value.lower().split()
    cut = max((i for i, w in enumerate(words) if w in _DETERMINERS), default=-1)
    words = words[cut + 1 :]
    while words and words[0].strip(":,") in _TASK_FILLER:
        words = words[1:]
    return " ".join(w.strip(":,") for w in words)


class _Pattern:
    def __init__(self, regex: str, predicate: str, confidence: float, *, negated: bool = False, kind: str = "") -> None:
        self.regex = re.compile(regex)
        self.predicate = predicate
        self.confidence = confidence
        self.negated = negated
        self.kind = kind  # "", "list", "date", "prefers", "category", "relation", "deadline", "task_free"


_I = r"\b(?i:i)"
_IM = r"\b(?i:i'?m|i\s+am)"

PATTERNS: list[_Pattern] = [
    # identity
    _Pattern(rf"\b(?i:my\s+name\s+is|call\s+me)\s+(?P<o>{PERSON})", "name", 0.95),
    _Pattern(r"^(?i:hi|hello|hey)[,!.]?\s+(?i:i'?m|i\s+am)\s+(?P<o>[A-Z][a-z]+)\b", "name", 0.85),
    _Pattern(rf"{_IM}\s+(?P<o>\d{{1,3}})\s+(?i:years\s+old)", "age", 0.9),
    _Pattern(rf"{_I}\s+(?i:just\s+)?(?i:turned)\s+(?P<o>\d{{1,3}})\b", "age", 0.85),
    _Pattern(rf"\b(?i:my\s+birthday\s+is)\s+(?i:on\s+)?(?P<o>{DATE})", "birthday", 0.9, kind="date"),
    # where the user lives
    _Pattern(
        rf"\b(?i:i|we)\s+(?i:(?:now|currently|still)\s+)?(?i:live|reside)\s+in\s+(?P<o>{PROPER})", "lives_in", 0.9
    ),
    _Pattern(
        rf"\b(?i:i|we)(?i:'ve|\s+have)?\s+(?i:(?:just|recently|finally|now)\s+)?(?i:moved|relocated)\s+"
        rf"(?i:(?:back|over)\s+)?(?i:to)\s+(?P<o>{PROPER})",
        "lives_in",
        0.9,
    ),
    _Pattern(rf"{_IM}\s+(?i:(?:now|currently)\s+)?(?i:based|living)\s+in\s+(?P<o>{PROPER})", "lives_in", 0.85),
    _Pattern(rf"{_I}(?i:'ve|\s+have)\s+(?i:been\s+living)\s+in\s+(?P<o>{PROPER})", "lives_in", 0.85),
    _Pattern(
        rf"\b(?i:i|we)\s+(?i:don't|do\s+not|no\s+longer)\s+(?i:live)\s+in\s+(?P<o>{PROPER})",
        "lives_in",
        0.85,
        negated=True,
    ),
    # employer
    _Pattern(rf"{_I}\s+(?i:(?:now|currently|still)\s+)?(?i:work)\s+(?i:at|for)\s+(?P<o>{PROPER})", "works_at", 0.85),
    _Pattern(
        rf"{_I}(?i:'ve|\s+have)?\s+(?i:(?:just|recently)\s+)?(?i:joined|started\s+at|started\s+working\s+at)\s+"
        rf"(?P<o>{PROPER})",
        "works_at",
        0.85,
    ),
    _Pattern(rf"\b(?i:my)\s+(?i:(?:new\s+)?(?:employer|company))\s+(?i:is)\s+(?P<o>{PROPER})", "works_at", 0.85),
    _Pattern(rf"{_I}\s+[^.;!?]*?\b(?i:and)\s+(?i:(?:now\s+)?work)\s+(?i:at|for)\s+(?P<o>{PROPER})", "works_at", 0.8),
    _Pattern(rf"{_I}\s+(?i:(?:now\s+)?work)\s+(?i:as)\s+[^.;!?]*?\s(?i:at|for)\s+(?P<o>{PROPER})", "works_at", 0.8),
    _Pattern(
        rf"{_I}\s+(?i:(?:just\s+)?(?:left|quit)|no\s+longer\s+work\s+(?:at|for))\s+(?P<o>{PROPER})",
        "works_at",
        0.8,
        negated=True,
    ),
    # job title
    _Pattern(
        rf"{_I}\s+(?:[^.;!?]*?\b(?i:and)\s+)?(?i:(?:now\s+)?work)\s+(?:(?i:at|for)\s+{PROPER}\s+)?(?i:as)\s+"
        r"(?i:an?\s+)?(?P<o>[A-Za-z][A-Za-z -]*?)"
        r"(?=\s+(?i:at|for|in|with|on)\b|[.,;!?]|$)",
        "job_title",
        0.8,
    ),
    _Pattern(
        r"\b(?i:my)\s+(?i:(?:new\s+)?(?:job\s+title|role|position|job))\s+(?i:is)\s+(?i:now\s+)?(?i:an?\s+)?"
        r"(?P<o>[A-Za-z][A-Za-z -]*?)(?=\s+(?i:at|for|in|with|on)\b|[.,;!?]|$)",
        "job_title",
        0.8,
    ),
    _Pattern(
        rf"{_IM}\s+(?i:(?:now|currently)\s+)?(?i:an?)\s+(?P<o>(?:[A-Za-z-]+\s+){{0,3}}?(?i:{JOB_NOUNS}))\b",
        "job_title",
        0.75,
    ),
    # manager
    _Pattern(
        r"\b(?i:my)\s+(?i:(?:new\s+)?(?:manager|boss|line\s+manager|supervisor|team\s+lead))\s+(?i:is)\s+"
        rf"(?i:(?:now|called|named)\s+)?(?P<o>{PERSON})",
        "manager",
        0.9,
    ),
    _Pattern(rf"{_I}\s+(?i:(?:now\s+)?report)\s+(?i:to)\s+(?P<o>{PERSON})", "manager", 0.85),
    _Pattern(
        rf"(?P<o>{PERSON})\s+(?i:is)\s+(?i:my)\s+(?i:(?:new\s+)?(?:manager|boss|line\s+manager|supervisor))",
        "manager",
        0.85,
    ),
    # relations and pets
    _Pattern(
        r"\b(?i:my)\s+(?P<rel>(?i:wife|husband|partner|girlfriend|boyfriend|dog|cat))(?:'s\s+name)?\s+(?i:is)\s+"
        rf"(?i:(?:called|named)\s+)?(?P<o>{PERSON})\b",
        "relation",
        0.85,
        kind="relation",
    ),
    _Pattern(
        rf"{_I}\s+(?i:have\s+an?)\s+(?P<rel>(?i:dog|cat))\s+(?i:called|named)\s+(?P<o>{PERSON})\b",
        "relation",
        0.85,
        kind="relation",
    ),
    # favourites and preferences
    _Pattern(
        r"\b(?i:my)\s+(?P<head>(?i:favou?rite|preferred))\s+(?P<c>[A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?i:is)\s+"
        rf"(?i:(?:now|still)\s+)?(?P<o>{PHRASE})",
        "category",
        0.85,
        kind="category",
    ),
    _Pattern(
        rf"{_I}(?i:'d|\s+would)?\s+(?i:(?:really|definitely|generally|usually|now|still|much)\s+)?(?i:prefer)\s+"
        r"(?P<o>[^.,;!?]+?)"
        r"(?=\s+(?i:over|to|rather\s+than|instead\s+of|if|when|because|since|unless|as)\s|[.,;!?]|$)",
        "prefers",
        0.75,
        kind="prefers",
    ),
    # likes, dislikes, allergies, languages
    _Pattern(
        rf"{_I}\s+(?i:(?:really|absolutely|also|still|totally)\s+)?(?i:like|love|enjoy|adore)\s+(?P<o>{PHRASE})",
        "likes",
        0.7,
        kind="list",
    ),
    _Pattern(
        rf"{_I}\s+(?i:(?:really|absolutely|also|still|totally)\s+)?"
        rf"(?i:don't\s+like|do\s+not\s+like|dislike|hate|can't\s+stand|cannot\s+stand|no\s+longer\s+like|"
        rf"don't\s+enjoy)\s+(?P<o>{PHRASE})",
        "dislikes",
        0.7,
        kind="list",
    ),
    _Pattern(
        rf"{_IM}\s+(?i:(?:severely|very|also|seriously)\s+)?(?i:allergic)\s+(?i:to)\s+(?P<o>{PHRASE})",
        "allergic_to",
        0.9,
        kind="list",
    ),
    _Pattern(rf"{_I}\s+(?i:have\s+an?)\s+(?P<o>[A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?i:allergy)", "allergic_to", 0.85),
    _Pattern(
        rf"{_I}\s+(?i:(?:also\s+)?speak)\s+(?P<o>[A-Z][a-z]+(?:(?:,\s*|\s+and\s+)[A-Z][a-z]+)*)",
        "speaks",
        0.8,
        kind="list",
    ),
    # deadlines
    _Pattern(
        rf"\b(?P<t>{TASK})\s+(?i:deadline|due\s+date)\s+{CHANGE}\s+(?i:(?:now|on|to)\s+)?(?P<o>{DATE})",
        "deadline",
        0.85,
        kind="deadline",
    ),
    _Pattern(
        rf"\b(?i:deadline|due\s+date)\s+(?i:for|of)\s+(?P<t>{TASK})\s+{CHANGE}\s+(?i:(?:now|on|to)\s+)?(?P<o>{DATE})",
        "deadline",
        0.85,
        kind="deadline",
    ),
    _Pattern(
        rf"\b(?P<t>{TASK})\s+(?i:is|are)\s+(?i:(?:now|still)\s+)?(?i:due)\s+(?i:(?:on|by)\s+)?(?P<o>{DATE})",
        "deadline",
        0.8,
        kind="deadline",
    ),
    _Pattern(
        rf"\b(?i:i|we)\s+(?i:need\s+to|have\s+to|must)\s+(?i:finish|submit|deliver|send|complete|hand\s+in)\s+"
        rf"(?P<t>{TASK})\s+(?i:by|before)\s+(?P<o>{DATE})",
        "deadline",
        0.8,
        kind="deadline",
    ),
    _Pattern(
        rf"\b(?i:the\s+)?(?i:deadline)\s+(?i:(?:has\s+(?:been\s+)?|was\s+|got\s+)?"
        rf"(?:moved|pushed|changed|shifted|postponed|brought\s+forward))\s+(?i:to)\s+(?P<o>{DATE})",
        "deadline:?",
        0.75,
        kind="date",
    ),
]


class FactExtractor(Protocol):
    """Turns sentences into fact candidates (one list per input sentence, so facts can be linked back)."""

    name: str

    def extract(self, sentences: Sequence[Sentence]) -> list[list[FactCandidate]]: ...


class RuleBasedExtractor:
    """Offline stand-in for an LLM extractor: deterministic first-person patterns."""

    name = "rules"

    def extract(self, sentences: Sequence[Sentence]) -> list[list[FactCandidate]]:
        return [self.extract_sentence(text, ts) for text, ts in sentences]

    def extract_sentence(self, text: str, ts: float | None = None) -> list[FactCandidate]:
        """All fact candidates in one sentence (empty for questions and hypotheticals)."""
        sentence = text.strip()
        if not sentence or _QUESTION.search(sentence) or _HYPOTHETICAL.search(sentence):
            return []
        correction = bool(_CORRECTION.search(sentence))
        found: list[FactCandidate] = []
        seen: set[tuple[str, str, bool]] = set()
        for pattern in PATTERNS:
            for match in pattern.regex.finditer(sentence):
                for predicate, raw in self._interpret(pattern, match):
                    obj = _clean_object(raw)
                    if not obj or len(obj) > 80 or _PRONOUN_OBJECT.match(obj):
                        continue
                    key = (predicate, obj.casefold(), pattern.negated)
                    if key in seen:
                        continue
                    seen.add(key)
                    confidence = min(0.99, pattern.confidence + (0.05 if correction else 0.0))
                    found.append(
                        FactCandidate(
                            predicate=predicate,
                            object=obj,
                            negated=pattern.negated,
                            confidence=confidence,
                            correction=correction,
                            sentence=sentence,
                            span=self._span(match, raw, obj),
                            ts=ts,
                        )
                    )
        if any(c.predicate.startswith("deadline:") and c.predicate != "deadline:?" for c in found):
            found = [c for c in found if c.predicate != "deadline:?"]
        return found

    @staticmethod
    def _span(match: re.Match[str], raw: str, obj: str) -> str:
        """Surface text of the object: the cleaned value if it occurs verbatim, else the raw regex group."""
        if obj.casefold() in match.string.casefold():
            return obj
        return _clean_object(match.group("o"))

    @staticmethod
    def _interpret(pattern: _Pattern, match: re.Match[str]) -> list[tuple[str, str]]:
        obj = match.group("o")
        kind = pattern.kind
        if kind == "relation":
            return [(normalise_predicate(match.group("rel")), obj)]
        if kind == "category":
            head = "favourite" if match.group("head").lower().startswith("fav") else "preferred"
            return [(normalise_predicate(f"{head}:{match.group('c')}"), obj)]
        if kind == "prefers":
            obj = _clean_object(re.sub(r"^(?:to\s+)", "", obj.strip(), flags=re.I))
            return [(f"prefers:{head_noun(obj)}", obj)]
        if kind == "deadline":
            task = _clean_task(match.group("t"))
            if not task:
                return []
            return [(normalise_predicate(f"deadline:{task}"), normalise_date(obj))]
        if kind == "date":
            return [(pattern.predicate, normalise_date(obj))]
        if kind == "list":
            obj = _CLAUSE_END.split(obj, maxsplit=1)[0]
            parts = [p.strip() for p in _LIST_SPLIT.split(_clean_object(obj)) if p.strip()]
            return [(pattern.predicate, p) for p in parts if len(p.split()) <= 4]
        return [(pattern.predicate, obj)]


class _LLMFact(BaseModel):
    sentence_index: int = Field(ge=0)
    subject: str = "user"
    predicate: str
    object: str
    negated: bool = False
    correction: bool = False
    confidence: float = Field(0.8, ge=0, le=1)


class _LLMFacts(BaseModel):
    facts: list[_LLMFact] = Field(default_factory=list)


_EXTRACT_PROMPT = """Extract durable facts about the user from the numbered sentences the user said.
Only first-person statements that are true now; skip questions, hypotheticals, jokes and facts about other people.
Use these predicates when they fit: name, lives_in, works_at, job_title, manager, age, birthday, partner, dog, cat,
likes, dislikes, allergic_to, speaks, deadline:<task>, favourite:<category>, preferred:<category>, prefers:<topic>.
Set negated=true for statements that something is no longer true ("I don't live in Rome any more").
Set correction=true when the user corrects something said earlier."""


class LLMExtractor:
    """Fact extraction by an LLM, with the rule-based extractor as the fallback when the call fails."""

    def __init__(self, llm: LLM, fallback: FactExtractor | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or RuleBasedExtractor()
        self.name = f"llm:{llm.name}"

    def extract(self, sentences: Sequence[Sentence]) -> list[list[FactCandidate]]:
        if not sentences:
            return []
        numbered = "\n".join(f"{i}. {text}" for i, (text, _) in enumerate(sentences))
        try:
            reply = self.llm.complete_json(
                [{"role": "system", "content": _EXTRACT_PROMPT}, {"role": "user", "content": numbered}], _LLMFacts
            )
        except LLMError:
            return self.fallback.extract(sentences)
        out: list[list[FactCandidate]] = [[] for _ in sentences]
        for fact in reply.facts:
            if fact.sentence_index >= len(sentences) or not fact.object.strip():
                continue
            text, ts = sentences[fact.sentence_index]
            predicate = normalise_predicate(fact.predicate)
            obj = normalise_date(fact.object) if predicate.startswith(("deadline", "birthday")) else fact.object
            out[fact.sentence_index].append(
                FactCandidate(
                    subject=fact.subject or "user",
                    predicate=predicate,
                    object=_clean_object(obj),
                    negated=fact.negated,
                    correction=fact.correction,
                    confidence=fact.confidence,
                    sentence=text,
                    ts=ts,
                )
            )
        return out
