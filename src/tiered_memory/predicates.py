"""Predicate vocabulary: canonical names, cardinality (one current value or many) and sentence rendering.

Cardinality drives contradiction handling. A single-valued predicate such as ``lives_in`` holds one current value,
so a new value supersedes the old one. A multi-valued predicate such as ``likes`` accumulates values; it only
conflicts with its opposite on the same object (``likes X`` versus ``dislikes X``).
"""

from __future__ import annotations

import re

SINGLE_VALUED = frozenset(
    {"name", "lives_in", "works_at", "job_title", "manager", "age", "birthday", "partner", "dog", "cat"}
)
SINGLE_VALUED_PREFIXES = ("deadline:", "favourite:", "preferred:", "prefers:")
OPPOSITES = {"likes": "dislikes", "dislikes": "likes"}

SYNONYMS = {
    "live_in": "lives_in",
    "lives": "lives_in",
    "resides_in": "lives_in",
    "based_in": "lives_in",
    "location": "lives_in",
    "city": "lives_in",
    "home_city": "lives_in",
    "employer": "works_at",
    "company": "works_at",
    "work_at": "works_at",
    "works_for": "works_at",
    "boss": "manager",
    "supervisor": "manager",
    "line_manager": "manager",
    "reports_to": "manager",
    "job": "job_title",
    "role": "job_title",
    "occupation": "job_title",
    "title": "job_title",
    "position": "job_title",
    "like": "likes",
    "loves": "likes",
    "enjoys": "likes",
    "dislike": "dislikes",
    "hates": "dislikes",
    "allergy": "allergic_to",
    "allergies": "allergic_to",
    "allergic": "allergic_to",
    "wife": "partner",
    "husband": "partner",
    "girlfriend": "partner",
    "boyfriend": "partner",
    "spouse": "partner",
    "favorite": "favourite",
    "preference": "prefers",
    "prefer": "prefers",
    "due": "deadline",
    "due_date": "deadline",
}

_ARTICLES = re.compile(r"^(?:the|a|an|my|our)\s+", re.I)
_NON_WORD = re.compile(r"[^\w\s:'-]")
_SPACES = re.compile(r"\s+")


def normalise_text(value: str) -> str:
    """Case-fold, drop punctuation and leading articles, collapse whitespace (used for equality checks)."""
    value = _NON_WORD.sub(" ", value.casefold())
    value = _SPACES.sub(" ", value).strip()
    return _ARTICLES.sub("", value)


def normalise_predicate(predicate: str) -> str:
    """Map free-form predicates (``"Resides in"``, ``"favorite:Food"``) onto the canonical vocabulary."""
    p = _SPACES.sub("_", predicate.strip().casefold().replace("-", "_"))
    p = p.replace("favorite", "favourite")
    if ":" in p:
        head, _, tail = p.partition(":")
        head = SYNONYMS.get(head, head)
        tail = normalise_text(tail.replace("_", " "))
        return f"{head}:{tail}" if tail else head
    return SYNONYMS.get(p, p)


def is_single_valued(predicate: str) -> bool:
    return predicate in SINGLE_VALUED or predicate.startswith(SINGLE_VALUED_PREFIXES)


def head_noun(phrase: str) -> str:
    """Crude head noun (last word, singularised) used to group preferences: 'window seats' -> 'seat'."""
    words = re.findall(r"[a-z]+", phrase.casefold())
    if not words:
        return "general"
    word = words[-1]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def _article(noun_phrase: str) -> str:
    return "an" if noun_phrase[:1].lower() in "aeiou" else "a"


def render_fact(subject: str, predicate: str, obj: str) -> str:
    """Natural-language sentence for a fact; this is what gets embedded and shown to the agent."""
    s = "The user" if subject.casefold() == "user" else subject
    poss = "The user's" if subject.casefold() == "user" else f"{subject}'s"
    simple = {
        "lives_in": f"{s} lives in {obj}.",
        "works_at": f"{s} works at {obj}.",
        "job_title": f"{s} works as {_article(obj)} {obj}.",
        "manager": f"{poss} manager is {obj}.",
        "name": f"{poss} name is {obj}.",
        "age": f"{s} is {obj} years old.",
        "birthday": f"{poss} birthday is {obj}.",
        "partner": f"{poss} partner is {obj}.",
        "dog": f"{poss} dog is called {obj}.",
        "cat": f"{poss} cat is called {obj}.",
        "likes": f"{s} likes {obj}.",
        "dislikes": f"{s} dislikes {obj}.",
        "allergic_to": f"{s} is allergic to {obj}.",
        "speaks": f"{s} speaks {obj}.",
    }
    if predicate in simple:
        return simple[predicate]
    head, _, tail = predicate.partition(":")
    if head == "deadline":
        return f"{poss} {tail} deadline is {obj}." if tail else f"{poss} deadline is {obj}."
    if head in ("favourite", "preferred"):
        return f"{poss} {head} {tail} is {obj}."
    if head == "prefers":
        return f"{s} prefers {obj}."
    return f"{s} {predicate.replace('_', ' ').replace(':', ' ')} {obj}."
