"""Seeded generator of synthetic multi-session conversations with facts that change over time.

Each simulated user has a profile (city, employer, manager, a project deadline, a meeting preference, a favourite
cuisine, an allergy, a dog and a conference talk). Over ``n_sessions`` sessions a few days apart, the user states these
facts, some of them change (moves, new jobs, new managers, moved deadlines), and most turns are distractor chatter:
small talk, questions, facts about friends ("my friend lives in Paris") and hypotheticals ("I might move to Berlin").
After the last session, probe questions ask for the *current* value of every slot; the earlier values are the stale
answers a memory should no longer return.

Phrasings are drawn from several templates per fact, including some that the rule-based extractor does not cover,
so extraction is imperfect by construction. Everything is deterministic for a given seed.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

ProbeKind = Literal["changed", "stable", "event"]

CITIES = [
    "Turin", "Milan", "Lyon", "Porto", "Munich", "Krakow", "Ghent", "Valencia", "Bologna", "Leeds", "Aarhus",
    "Utrecht", "Graz", "Brno", "Tallinn", "Bilbao", "Seville", "Antwerp", "Dresden", "Gothenburg", "Lille", "Bergen",
    "Cork", "Salzburg", "Zagreb", "Ljubljana", "Basel", "Nantes", "Malaga", "Hamburg",
]  # fmt: skip
OTHER_CITIES = ["Paris", "Rome", "Berlin", "Madrid", "Vienna", "Prague", "Lisbon", "Dublin", "Oslo", "Athens"]
COMPANIES = [
    "Northwind Labs", "Bluefin Analytics", "Helix Robotics", "Quarry Systems", "Lumen Health", "Arbor Logistics",
    "Copperleaf Energy", "Tessellate", "Brightwater Bank", "Orbital Foods", "Kestrel Media", "Juniper Mobility",
    "Silverline Insurance", "Mosaic Learning", "Pinecrest Pharma", "Redwood Retail",
]  # fmt: skip
OTHER_COMPANIES = ["Globex", "Initech", "Umbrella Corp", "Vandelay Industries", "Stark Industries"]
PEOPLE = [
    "Paolo Ferri", "Giulia Conti", "Anna Rossi", "Marta Klein", "Jonas Weber", "Elena Petrova", "Tomas Novak",
    "Ines Duarte", "Sofia Lund", "Karim Haddad", "Lucia Moreno", "Hannah Becker", "Omar Said", "Clara Nilsson",
    "Felix Wagner", "Nadia Kowalski", "Ruben Silva", "Eva Horvat", "Luca Bianchi", "Maya Fischer",
]  # fmt: skip
FRIENDS = ["Marco", "Chiara", "Ben", "Lena", "Diego", "Aisha", "Tom", "Yuki", "Pablo", "Freya"]
PROJECTS = [
    "quarterly report", "Atlas migration", "budget review", "grant proposal", "website relaunch",
    "audit preparation", "pricing study", "vendor evaluation",
]  # fmt: skip
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "September", "October", "November"]
MEETINGS = ["morning meetings", "afternoon meetings", "short stand-up meetings", "walking meetings", "video meetings"]
CUISINES = ["Thai", "Ethiopian", "Georgian", "Peruvian", "Lebanese", "Korean", "Vietnamese", "Mexican"]
OTHER_CUISINES = ["Italian", "Greek", "Indian", "Chinese", "French", "Spanish"]
ALLERGIES = ["peanuts", "shellfish", "penicillin", "pollen", "sesame", "latex"]
DOGS = ["Biscuit", "Pepper", "Juno", "Ollie", "Maple", "Rocket", "Nala", "Ziggy", "Pixel", "Mochi"]
CONFERENCES = ["DataSummit", "CloudConf", "PyData", "Web Forward", "AI Frontiers", "Product Week", "DevDays"]

# Statement templates. Entries marked with a trailing "~" are phrasings the rule-based extractor is not expected to
# handle (they still count in the extraction metrics, which is the point).
INITIAL = {
    "lives_in": ["I live in {v}.", "I'm based in {v}.", "I've been living in {v} for a few years now.",
                 "Home for me is {v}.~", "I live in {v}, close to the old town."],
    "works_at": ["I work at {v}.", "My company is {v}.", "I work for {v} on the data team.",
                 "I've been at {v} for two years.~"],
    "manager": ["My manager is {v}.", "I report to {v}.", "{v} is my manager.", "My boss, {v}, is very supportive.~"],
    "deadline": ["The {p} deadline is {v}.", "The {p} is due on {v}.", "I need to finish the {p} by {v}.",
                 "The deadline for the {p} is {v}."],
    "prefers:meeting": ["I prefer {v}.", "I'd prefer {v} if possible.", "Please book {v} for me, they work best.~"],
    "favourite:cuisine": ["My favourite cuisine is {v}.", "My favorite cuisine is {v} food.",
                          "Nothing beats {v} food for me.~"],
    "allergic_to": ["Remember that I'm allergic to {v}.", "I'm allergic to {v}, so please keep that in mind.",
                    "Just so you know, {v} make me really ill.~"],
    "dog": ["My dog is called {v}.", "I have a dog named {v}.", "{v}, my dog, chewed my charger again.~"],
    "event": ["Remember that I'm presenting at {v} on {d}.", "I'm giving a talk at {v} on {d}, slides due soon.",
              "Next month I speak at {v}, quite nervous about it.~"],
}  # fmt: skip
UPDATE = {
    "lives_in": ["I just moved to {v} last weekend.", "Big news: we relocated to {v} this month.",
                 "I now live in {v}.", "I've moved to {v}, still unpacking boxes.", "These days home is {v}.~"],
    "works_at": ["I joined {v} this week.", "I started at {v} on Monday.", "I left my old job and now work for {v}.",
                 "New job: I'm at {v} now.~"],
    "manager": ["My new manager is {v}.", "I now report to {v}.", "{v} is my new manager.",
                "After the reorg {v} runs my team.~"],
    "deadline": ["Actually, the {p} deadline moved to {v}.", "The deadline for the {p} has been pushed to {v}.",
                 "Update: the {p} is now due on {v}.", "They moved the {p} to {v}.~"],
    "prefers:meeting": ["I prefer {v} now.", "From now on I'd prefer {v}.", "Please switch me to {v}.~"],
    "favourite:cuisine": ["My favourite cuisine is now {v}.", "My favorite cuisine is {v} these days.",
                          "Lately I'm obsessed with {v} food.~"],
}  # fmt: skip
QUESTIONS = {
    "lives_in": ["Which city does the user live in?", "Where is the user based these days?"],
    "works_at": ["Which company does the user work for?", "Who is the user's employer?"],
    "manager": ["Who is the user's manager?", "Who does the user report to?"],
    "deadline": ["When is the {p} due?", "What is the deadline for the {p}?"],
    "prefers:meeting": ["What kind of meetings does the user prefer?", "How does the user like meetings scheduled?"],
    "favourite:cuisine": ["What is the user's favourite cuisine?", "Which food does the user like most?"],
    "allergic_to": ["Does the user have any allergies?", "What is the user allergic to?"],
    "dog": ["What is the name of the user's dog?", "What is the user's dog called?"],
    "event": ["Which conference is the user presenting at?", "Where is the user giving a talk?"],
}  # fmt: skip
CHATTER = [
    "The weather has been {adj} all week.", "Did you see the match last night? It ended {a}-{b}.",
    "I had a great {oc} lunch today.", "My friend {f} lives in {oc_city}.", "My sister just moved to {oc_city}.",
    "My colleague {f} works at {o_company}.", "I might move to {oc_city} one day.",
    "If I lived in {oc_city} I would cycle everywhere.", "Can you recommend a podcast about history?",
    "What's a quick recipe for dinner tonight?", "How do I convert {n} euros to pounds?",
    "Can you draft an email to the team about Friday's lunch?", "Remind me what a pivot table is.",
    "I watched a documentary about volcanoes yesterday.", "Traffic was terrible this morning.",
    "My friend {f}'s manager is {o_person}.", "I'm thinking of learning to play the guitar.",
    "We booked a weekend trip to {oc_city} for {n} euros.", "Ok, thanks!", "Haha, that's funny.",
]  # fmt: skip
REPLIES = [
    "Sure, here is a short answer.", "Thanks for letting me know.", "Got it.", "Happy to help with that.",
    "That sounds lovely.", "Here are a few ideas you could try.", "Noted.", "Interesting! Tell me more.",
]  # fmt: skip
ECHO = {
    "lives_in": "Got it, you're in {v} now.",
    "works_at": "Congratulations on {v}!",
    "manager": "Noted, {v} is your manager.",
    "deadline": "Okay, {v} for the {p}.",
    "allergic_to": "Understood, I'll remember the {v} allergy.",
}
REMINDERS = {
    "lives_in": ["Still enjoying life in {v}.", "It's been raining in {v} all day."],
    "works_at": ["Busy week at {v}.", "We had a team lunch at {v} today."],
    "manager": ["Had a good one-to-one with {v} today."],
}


class FactMention(BaseModel):
    """Ground truth: a turn states that ``slot`` has ``value``."""

    slot: str
    value: str
    covered: bool = Field(description="False for phrasings the rule-based extractor is not designed to handle.")


class SimTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    ts: float
    mentions: list[FactMention] = Field(default_factory=list)


class SimSession(BaseModel):
    index: int
    start_ts: float
    turns: list[SimTurn]


class Probe(BaseModel):
    user_id: str
    slot: str
    kind: ProbeKind
    question: str
    gold: str
    stale: list[str] = Field(default_factory=list)
    all_covered: bool = Field(
        True, description="Every statement of this fact used a phrasing the rule-based extractor is designed for."
    )


class SimUser(BaseModel):
    user_id: str
    sessions: list[SimSession]
    probes: list[Probe]


class Dataset(BaseModel):
    seed: int
    n_users: int
    n_sessions: int
    users: list[SimUser]

    @property
    def n_turns(self) -> int:
        return sum(len(s.turns) for u in self.users for s in u.sessions)


def _date(rng: random.Random, exclude: set[str]) -> str:
    while True:
        value = f"{rng.randint(1, 28)} {rng.choice(MONTHS)}"
        if value not in exclude:
            return value


def _fill(template: str, rng: random.Random, **values: str) -> tuple[str, bool]:
    covered = not template.endswith("~")
    text = template.rstrip("~").format(**values)
    return text[0].upper() + text[1:], covered


def _chatter(rng: random.Random) -> str:
    template = rng.choice(CHATTER)
    return template.format(
        adj=rng.choice(["grey", "sunny", "windy", "cold", "humid"]),
        a=rng.randint(0, 4),
        b=rng.randint(0, 4),
        oc=rng.choice(OTHER_CUISINES),
        f=rng.choice(FRIENDS),
        oc_city=rng.choice(OTHER_CITIES),
        o_company=rng.choice(OTHER_COMPANIES),
        o_person=rng.choice(["Karl Berg", "Rosa Lind", "Ivan Toth"]),
        n=rng.randint(20, 400),
    )


def _user(user_index: int, n_sessions: int, seed: int, start: float) -> SimUser:
    rng = random.Random(seed * 100_003 + user_index)
    user_id = f"user-{user_index:03d}"
    project = rng.choice(PROJECTS)

    # Value timelines: slot -> list of (session index, value). Changes happen from session 2 on.
    def timeline(pool: list[str], p_change: float, max_changes: int) -> list[tuple[int, str]]:
        values = rng.sample(pool, k=1 + max_changes)
        changes = 0
        if rng.random() < p_change:
            changes = rng.randint(1, max_changes)
        sessions = sorted(rng.sample(range(2, n_sessions), k=min(changes, max(0, n_sessions - 2))))
        first = rng.choice([0, 0, 1]) if n_sessions > 1 else 0
        return [(first, values[0])] + [(s, values[i + 1]) for i, s in enumerate(sessions)]

    timelines: dict[str, list[tuple[int, str]]] = {
        "lives_in": timeline(CITIES, 0.7, 2),
        "works_at": timeline(COMPANIES, 0.5, 1),
        "manager": timeline(PEOPLE, 0.6, 2),
        "prefers:meeting": timeline(MEETINGS, 0.5, 1),
        "favourite:cuisine": timeline(CUISINES, 0.4, 1),
    }
    used_dates: set[str] = set()
    deadline_values = []
    for _ in range(3):
        deadline_values.append(_date(rng, used_dates))
        used_dates.add(deadline_values[-1])
    n_dead_changes = rng.choice([0, 1, 1, 2, 2]) if n_sessions > 2 else 0
    dead_sessions = sorted(rng.sample(range(2, n_sessions), k=min(n_dead_changes, max(0, n_sessions - 2))))
    timelines["deadline"] = [(rng.choice([0, 1]) if n_sessions > 1 else 0, deadline_values[0])] + [
        (s, deadline_values[i + 1]) for i, s in enumerate(dead_sessions)
    ]
    stable = {
        "allergic_to": (rng.randrange(0, min(3, n_sessions)), rng.choice(ALLERGIES)),
        "dog": (rng.randrange(0, min(3, n_sessions)), rng.choice(DOGS)),
        "event": (rng.randrange(0, n_sessions), rng.choice(CONFERENCES)),
    }
    event_date = _date(rng, used_dates)

    sessions: list[SimSession] = []
    t = start + rng.uniform(0, 2) * 86400
    for s_idx in range(n_sessions):
        statements: list[tuple[str, str, str, bool]] = []  # (slot, value, text, covered)
        for slot, line in timelines.items():
            for when, value in line:
                if when != s_idx:
                    continue
                templates = INITIAL[slot] if when == line[0][0] else UPDATE[slot]
                text, covered = _fill(rng.choice(templates), rng, v=value, p=project)
                statements.append((slot, value, text, covered))
        for slot, (when, value) in stable.items():
            if when == s_idx:
                text, covered = _fill(rng.choice(INITIAL[slot]), rng, v=value, d=event_date)
                statements.append((slot, value, text, covered))
        reminders: list[str] = []
        for slot, line in timelines.items():
            current = [v for when, v in line if when <= s_idx]
            if slot in REMINDERS and current and rng.random() < 0.25:
                reminders.append(rng.choice(REMINDERS[slot]).format(v=current[-1]))
        n_chatter = rng.randint(4, 7)
        user_lines: list[tuple[str, list[FactMention], str | None]] = [
            (text, [FactMention(slot=slot, value=value, covered=covered)], slot)
            for slot, value, text, covered in statements
        ]
        user_lines += [(text, [], None) for text in reminders]
        user_lines += [(_chatter(rng), [], None) for _ in range(n_chatter)]
        rng.shuffle(user_lines)

        turns: list[SimTurn] = []
        for text, mentions, stated in user_lines:
            t += rng.uniform(40, 150)
            turns.append(SimTurn(role="user", text=text, ts=t, mentions=mentions))
            t += rng.uniform(5, 20)
            if stated in ECHO and rng.random() < 0.35:
                reply = ECHO[stated].format(v=mentions[0].value, p=project)
            else:
                reply = rng.choice(REPLIES)
            turns.append(SimTurn(role="assistant", text=reply, ts=t))
        sessions.append(SimSession(index=s_idx, start_ts=turns[0].ts, turns=turns))
        t += rng.uniform(3, 12) * 86400

    def all_covered(slot: str) -> bool:
        return all(m.covered for s in sessions for t in s.turns for m in t.mentions if m.slot == slot)

    probes: list[Probe] = []
    for slot, line in timelines.items():
        values = [v for _, v in line]
        question = rng.choice(QUESTIONS[slot]).format(p=project)
        probes.append(
            Probe(
                user_id=user_id,
                slot=slot,
                kind="changed" if len(values) > 1 else "stable",
                question=question,
                gold=values[-1],
                stale=[v for v in values[:-1] if v != values[-1]],
                all_covered=all_covered(slot),
            )
        )
    for slot, (_, value) in stable.items():
        probes.append(
            Probe(
                user_id=user_id,
                slot=slot,
                kind="event" if slot == "event" else "stable",
                question=rng.choice(QUESTIONS[slot]),
                gold=value,
                all_covered=all_covered(slot),
            )
        )
    return SimUser(user_id=user_id, sessions=sessions, probes=probes)


def generate(
    n_users: int = 50,
    n_sessions: int = 10,
    seed: int = 7,
    start: datetime = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc),
) -> Dataset:
    """Generate ``n_users`` synthetic users with ``n_sessions`` sessions each (deterministic for a seed)."""
    if n_users < 1 or n_sessions < 1:
        raise ValueError("n_users and n_sessions must be at least 1")
    users = [_user(i, n_sessions, seed, start.timestamp()) for i in range(n_users)]
    return Dataset(seed=seed, n_users=n_users, n_sessions=n_sessions, users=users)
