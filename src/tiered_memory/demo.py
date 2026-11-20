"""Scripted three-session demo: what gets remembered, compressed, superseded and evicted."""

from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .bench.baselines import FlatVectorStore
from .clock import ManualClock
from .config import MemoryConfig
from .embeddings import Embedder
from .memory import MemorySystem
from .models import ConsolidationReport, FactChange, fmt_day

SESSIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Session 1 · Monday 2 March 2026",
        [
            ("user", "Hi! I'm Sara. I live in Turin and work at Fiat as a data analyst."),
            ("assistant", "Nice to meet you, Sara! What can I do for you today?"),
            ("user", "Remember that I'm allergic to peanuts."),
            ("assistant", "Noted: no peanuts, ever."),
            ("user", "The weather is lovely today, finally some sun."),
            ("assistant", "Enjoy it!"),
            ("user", "My manager is Paolo Ferri and the quarterly report deadline is 20 March."),
            ("assistant", "Got it, I will keep the deadline in mind."),
            ("user", "I prefer morning meetings."),
        ],
    ),
    (
        "Session 2 · Wednesday 11 March 2026",
        [
            ("user", "Quick update: I just moved to Milan last weekend."),
            ("assistant", "Congratulations on the move!"),
            ("user", "Did you see the match last night? It ended 2-1."),
            ("assistant", "I did not, but it sounds like a good one."),
            ("user", "Actually, the quarterly report deadline moved to 27 March."),
            ("user", "My new manager is Giulia Conti."),
            ("user", "I'm presenting the report at the Lisbon offsite on 2 April."),
        ],
    ),
]
QUESTIONS = [
    "Where does Sara live?",
    "Does Sara have any food allergies?",
    "Where is Sara presenting the report?",
]
DEMO_CONFIG = MemoryConfig(working_budget_tokens=60, episode_ttl_days=14)
START = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)

_ACTION_STYLE = {
    "added": "green",
    "superseded": "bold yellow",
    "reinforced": "cyan",
    "merged": "cyan",
    "retracted": "red",
    "historical": "magenta",
    "blocked": "red",
    "ignored": "dim",
}


def _importance_bar(value: float) -> Text:
    filled = round(value * 10)
    style = "bold red" if value >= 0.7 else "yellow" if value >= 0.4 else "dim"
    return Text("█" * filled + "·" * (10 - filled), style=style)


def _fact_rows(changes: list[FactChange], table: Table) -> None:
    for change in changes:
        style = _ACTION_STYLE.get(change.action, "")
        detail = f"replaces: {change.replaced_text}" if change.replaced_text else ""
        table.add_row(Text(change.action, style=style), Text(change.text), Text(detail, style="dim"))


def _consolidation_line(report: ConsolidationReport) -> Text:
    share = report.kept_tokens / report.raw_tokens if report.raw_tokens else 0.0
    trigger = "working memory full" if report.reason == "overflow" else "session ended"
    return Text.assemble(
        (f"  ⤷ {trigger}: ", "bold magenta"),
        f"{report.sentences_in} sentences, {report.raw_tokens} tokens → episode {report.episode_id} ",
        (f"(kept {report.sentences_kept}, {report.kept_tokens} tokens = {share:.0%})", "magenta"),
    )


def run_demo(console: Console, embedder: Embedder) -> MemorySystem:
    """Play the scripted sessions against a fresh in-memory store and narrate every memory operation."""
    clock = ManualClock(START)
    memory = MemorySystem(namespace="sara", config=DEMO_CONFIG, embedder=embedder, clock=clock, offline=True)
    flat = FlatVectorStore(embedder)
    console.print(
        Panel.fit(
            Text.assemble(
                ("tiered-memory", "bold"),
                " · working scratchpad → compressed episodes → versioned facts\n",
                (f"embedder: {embedder.name} · components: offline (TextRank, rules, heuristics) · "
                 f"working budget: {DEMO_CONFIG.working_budget_tokens} tokens", "dim"),
            ),
            border_style="blue",
        )
    )
    for index, (title, turns) in enumerate(SESSIONS):
        if index:
            clock.set(START.timestamp() + 9 * 86400)
        sid = memory.start_session(f"s{index + 1}")
        console.rule(f"[bold]{title}")
        flushes: list[ConsolidationReport] = []
        for role, text in turns:
            clock.advance(minutes=2)
            result = memory.observe(sid, role, text)
            flat.observe("sara", index, role, text, clock.now())
            line = Table.grid(padding=(0, 1), expand=True)
            line.add_column(width=9, justify="right")
            line.add_column(width=15)
            line.add_column(ratio=3)
            line.add_column(ratio=1)
            reasons = ", ".join(r for r in result.importance.reasons if r != "assistant")
            line.add_row(
                Text(role, style="bold blue" if role == "user" else "dim"),
                Text.assemble(_importance_bar(result.importance.value), f" {result.importance.value:.2f}"),
                Text(text, style="" if role == "user" else "dim"),
                Text(reasons, style="dim"),
            )
            console.print(line)
            if result.consolidation and result.consolidation.episode_id:
                console.print(_consolidation_line(result.consolidation))
                flushes.append(result.consolidation)
        clock.advance(minutes=5)
        flat.end_session("sara", index, clock.now())
        final = memory.end_session(sid)
        if final.episode_id:
            console.print(_consolidation_line(final))
        table = Table(title="semantic memory changes", title_justify="left", show_header=True, header_style="bold",
                      expand=True)
        table.add_column("action", width=11)
        table.add_column("fact")
        table.add_column("", ratio=1)
        for report in [*flushes, final]:
            _fact_rows(report.facts, table)
        console.print(table)

    clock.set(START.timestamp() + 40 * 86400)
    console.rule("[bold]Session 3 · Saturday 11 April 2026 (40 days later)")
    maintenance = memory.maintain()
    evicted = [t for t in memory.tombstones() if t.deleted_at == maintenance.now]
    console.print(
        Text.assemble(
            ("maintenance: ", "bold"),
            f"{len(maintenance.expired)} expired by TTL, {len(maintenance.evicted)} evicted by capacity · ",
            (", ".join(f"{t.id} ({t.label}, {t.reason})" for t in evicted) or "nothing to evict", "dim"),
        )
    )
    survivors = Table(title="episodes still in memory (outdated values masked)", title_justify="left", expand=True,
                      header_style="bold")
    survivors.add_column("episode", width=15)
    survivors.add_column("importance", width=10)
    survivors.add_column("expires", width=10)
    survivors.add_column("kept sentences", ratio=1)
    current = {f.id for f in memory.facts()}
    for episode in memory.episodes():
        expires = "never" if episode.expires_at is None else fmt_day(episode.expires_at)
        hidden = set(episode.fact_ids) - current
        survivors.add_row(episode.id, f"{episode.importance:.2f}", expires, Text(episode.render(hidden)))
    console.print(survivors)

    for question in QUESTIONS:
        recalled = memory.recall(question, budget_tokens=60, k=3)
        table = Table(title=f"recall: {question!r}  (budget 60 tokens)", title_justify="left", expand=True,
                      header_style="bold")
        table.add_column("tier", width=8)
        table.add_column("score", width=5, justify="right")
        table.add_column("rel · imp · rec", width=15)
        table.add_column("memory", ratio=1)
        table.add_column("provenance", width=24)
        for item in recalled.items:
            prov = item.provenance
            if item.tier == "semantic":
                where = f"fact since {fmt_day(prov.valid_from)}"
            else:
                where = f"{prov.source_episode} ({prov.session_id})"
            s = item.score
            table.add_row(item.tier, f"{s.total:.2f}", f"{s.relevance:.2f}·{s.importance:.2f}·{s.recency:.2f}",
                          Text(item.text), Text(where))
        flat_top = flat.query("sara", question, k=1, budget_tokens=10**6)
        console.print(Group(table, Text.assemble(("  a flat vector store over all turns would answer: ", "dim"),
                                                 (flat_top[0] if flat_top else "-", "italic"))))

    history = Table(title="history of the user's city (validity intervals)", title_justify="left",
                    header_style="bold")
    for column in ("fact", "valid from", "valid to", "status"):
        history.add_column(column)
    for fact in memory.history("lives_in"):
        history.add_row(Text(fact.text), fmt_day(fact.valid_from), fmt_day(fact.valid_to),
                        Text(fact.status, style="green" if fact.status == "current" else "yellow"))
    console.print(history)
    stats = memory.stats()
    console.print(
        Text.assemble(
            ("store: ", "bold"),
            f"{stats.facts_current} current facts, {stats.facts_history} historical, {stats.episodes} episodes, "
            f"{stats.tombstones} tombstones, ~{stats.approx_bytes / 1024:.1f} KB",
        )
    )
    return memory
