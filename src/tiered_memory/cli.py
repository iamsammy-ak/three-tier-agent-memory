"""Command-line interface: ``tiered-memory`` (or ``python -m tiered_memory``)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .clock import Clock, ManualClock, SystemClock, parse_when
from .embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder, get_embedder
from .memory import MemorySystem
from .models import ConsolidationReport, fmt_day, fmt_ts
from .store import EmbedderMismatchError, SQLiteStore

app = typer.Typer(
    help="Three-tier long-term memory for LLM agents: working scratchpad, compressed episodes, versioned facts.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

DEFAULT_DB = os.environ.get("TIERED_MEMORY_DB", ".tiered-memory/memory.db")
DbOpt = Annotated[Path, typer.Option("--db", help="SQLite file (env TIERED_MEMORY_DB).")]
UserOpt = Annotated[str, typer.Option("--user", "-u", help="Namespace: one user or agent.")]
EmbedderOpt = Annotated[str, typer.Option("--embedder", help="auto | fastembed | hashing")]
OfflineOpt = Annotated[bool, typer.Option("--offline", help="Never call an LLM, even if a key is set.")]
NowOpt = Annotated[str | None, typer.Option("--now", help="Pretend it is this ISO date-time (UTC).")]


def _embedder_for(store: SQLiteStore, kind: str) -> Embedder:
    """Reuse the embedder a store was built with when ``auto`` is requested."""
    signature = store.get_meta("embedder")
    if kind == "auto" and signature:
        name = signature.split("|")[0]
        if name.startswith("fastembed:"):
            return FastEmbedEmbedder(name.split(":", 1)[1])
        if name.startswith("hashing-"):
            return HashingEmbedder(int(name.split("-")[1]))
    return get_embedder(kind)


def _open(db: Path, user: str, embedder: str, offline: bool, now: str | None = None) -> MemorySystem:
    store = SQLiteStore(db)
    clock: Clock = ManualClock(parse_when(now)) if now else SystemClock()
    try:
        return MemorySystem(store, namespace=user, embedder=_embedder_for(store, embedder), clock=clock,
                            offline=offline)
    except EmbedderMismatchError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


def _print_report(report: ConsolidationReport) -> None:
    if report.episode_id:
        console.print(
            f"episode [bold]{report.episode_id}[/bold]: {report.sentences_kept}/{report.sentences_in} sentences, "
            f"{report.kept_tokens}/{report.raw_tokens} tokens kept"
        )
    if report.facts:
        table = Table("action", "fact", "replaces", header_style="bold")
        for change in report.facts:
            table.add_row(change.action, Text(change.text), Text(change.replaced_text or ""))
        console.print(table)
    elif not report.episode_id:
        console.print("[dim]nothing stored[/dim]")


@app.command()
def demo(
    svg: Annotated[Path | None, typer.Option(help="Also save the output as an SVG terminal recording.")] = None,
    embedder: EmbedderOpt = "auto",
    width: Annotated[int, typer.Option(help="Console width in columns.")] = 100,
) -> None:
    """Three scripted sessions: what is remembered, compressed, superseded and evicted (offline, in memory)."""
    from .demo import run_demo

    out = Console(record=svg is not None, width=width)
    run_demo(out, get_embedder(embedder))
    if svg is not None:
        svg.parent.mkdir(parents=True, exist_ok=True)
        out.save_svg(str(svg), title="tiered-memory demo")
        svg.write_text(re.sub(r"\s*<!--.*?-->", "", svg.read_text(encoding="utf-8"), flags=re.S), encoding="utf-8")
        console.print(f"[dim]saved {svg}[/dim]")


@app.command()
def remember(
    text: Annotated[str, typer.Argument(help="Something to remember, e.g. \"I'm allergic to peanuts\".")],
    user: UserOpt = "default",
    db: DbOpt = Path(DEFAULT_DB),
    embedder: EmbedderOpt = "auto",
    offline: OfflineOpt = False,
) -> None:
    """Store an explicit memory now: a protected note plus any facts it states."""
    _print_report(_open(db, user, embedder, offline).remember(text))


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="JSONL transcript: {session, role, text, ts?} per line.")],
    user: UserOpt = "default",
    db: DbOpt = Path(DEFAULT_DB),
    embedder: EmbedderOpt = "auto",
    offline: OfflineOpt = False,
) -> None:
    """Replay a transcript through working memory and consolidate every session in it."""
    memory = _open(db, user, embedder, offline)
    sessions: list[str] = []
    turns = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            session, role, text = str(row["session"]), str(row["role"]), str(row["text"])
        except (json.JSONDecodeError, KeyError) as exc:
            console.print(f"[red]line {line_no}: expected {{session, role, text, ts?}} ({exc})[/red]")
            raise typer.Exit(2) from exc
        ts = row.get("ts")
        when = parse_when(ts) if isinstance(ts, str) else (float(ts) if ts is not None else None)
        if session not in sessions:
            sessions.append(session)
            memory.start_session(session)
        memory.observe(session, role, text, ts=when)
        turns += 1
    for session in sessions:
        memory.end_session(session)
    stats = memory.stats()
    console.print(
        f"ingested {turns} turns in {len(sessions)} session(s) → {stats.episodes} episodes, "
        f"{stats.facts_current} current facts, {stats.facts_history} historical"
    )


@app.command()
def recall(
    query: Annotated[str, typer.Argument(help="What the agent needs to know.")],
    user: UserOpt = "default",
    db: DbOpt = Path(DEFAULT_DB),
    budget: Annotated[int, typer.Option(help="Token budget for the packed context.")] = 256,
    k: Annotated[int | None, typer.Option(help="Maximum number of items.")] = None,
    history: Annotated[bool, typer.Option(help="Include superseded facts and unredacted episodes.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full result as JSON.")] = False,
    embedder: EmbedderOpt = "auto",
    offline: OfflineOpt = False,
    now: NowOpt = None,
) -> None:
    """Recall memories for a query across all tiers, packed into a token budget, with scores and provenance."""
    result = _open(db, user, embedder, offline, now).recall(query, budget_tokens=budget, k=k, include_history=history)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    table = Table(title=f"{query!r} · {result.used_tokens}/{budget} tokens · {result.considered} candidates",
                  header_style="bold", expand=True)
    table.add_column("tier", width=9)
    table.add_column("score", justify="right", width=6)
    table.add_column("rel·imp·rec", width=14)
    table.add_column("memory", ratio=1)
    table.add_column("provenance", width=28)
    for item in result.items:
        p, s = item.provenance, item.score
        where = (
            f"{fmt_day(p.valid_from)} → {fmt_day(p.valid_to)} ({p.status})"
            if item.tier == "semantic"
            else f"{p.session_id} · {p.note or ''}"
        )
        table.add_row(item.tier, f"{s.total:.2f}", f"{s.relevance:.2f}·{s.importance:.2f}·{s.recency:.2f}",
                      Text(item.text), Text(where))
    console.print(table)
    if result.items:
        console.print(Text("context block for the prompt:", style="dim"))
        console.print(Text(result.as_context()))


@app.command()
def consolidate(
    user: UserOpt = "default",
    db: DbOpt = Path(DEFAULT_DB),
    embedder: EmbedderOpt = "auto",
    offline: OfflineOpt = False,
    now: NowOpt = None,
) -> None:
    """Consolidate idle sessions into episodes and facts, then apply TTL expiry and capacity eviction."""
    report = _open(db, user, embedder, offline, now).maintain()
    for item in report.consolidated:
        console.print(f"[bold]session {item.session_id}[/bold] ({item.reason})")
        _print_report(item)
    console.print(
        f"{len(report.consolidated)} idle session(s) consolidated · {len(report.expired)} expired · "
        f"{len(report.evicted)} evicted (as of {fmt_ts(report.now)} UTC)"
    )


@app.command()
def inspect(
    user: UserOpt = "default",
    db: DbOpt = Path(DEFAULT_DB),
    tier: Annotated[str, typer.Option(help="all | facts | episodes | working | tombstones | events")] = "all",
    history: Annotated[bool, typer.Option(help="Include superseded and retracted facts.")] = False,
    embedder: EmbedderOpt = "auto",
) -> None:
    """Show what a namespace remembers, tier by tier."""
    memory = _open(db, user, embedder, offline=True)
    stats = memory.stats()
    console.print(
        f"[bold]{user}[/bold] in {db}: {stats.facts_current} current facts, {stats.facts_history} historical, "
        f"{stats.episodes} episodes, {stats.working_items} working items, {stats.tombstones} tombstones · "
        f"namespaces: {', '.join(memory.store.namespaces()) or '-'}"
    )
    if tier in ("all", "facts"):
        table = Table("id", "fact", "conf", "imp", "valid from", "valid to", "status", title="semantic memory",
                      header_style="bold", title_justify="left")
        for f in memory.facts(include_history=history):
            table.add_row(f.id, Text(f.text), f"{f.confidence:.2f}", f"{f.importance:.2f}", fmt_day(f.valid_from),
                          fmt_day(f.valid_to), f.status)
        console.print(table)
    if tier in ("all", "episodes"):
        current = {f.id for f in memory.facts()}
        table = Table("id", "session", "when", "imp", "tokens", "expires", "text (outdated values hidden)",
                      title="episodic memory", header_style="bold", title_justify="left")
        for e in memory.episodes():
            hidden = set(e.fact_ids) - current
            expires = "never" if e.expires_at is None else fmt_day(e.expires_at)
            table.add_row(e.id, e.session_id, fmt_ts(e.end_ts), f"{e.importance:.2f}",
                          f"{e.tokens}/{e.raw_tokens}", expires, Text(e.render(hidden)))
        console.print(table)
    if tier in ("all", "working"):
        table = Table("session", "kind", "role", "text", "tokens", "imp", title="working memory",
                      header_style="bold", title_justify="left")
        for w in memory.working_items():
            table.add_row(w.session_id, w.kind, w.role, Text(w.text), str(w.tokens), f"{w.importance:.2f}")
        console.print(table)
    if tier in ("all", "tombstones"):
        table = Table("id", "tier", "label", "reason", "deleted", title="tombstones", header_style="bold",
                      title_justify="left")
        for t in memory.tombstones():
            table.add_row(t.id, t.tier, t.label, t.reason, fmt_ts(t.deleted_at))
        console.print(table)
    if tier == "events":
        for event in memory.events(100):
            console.print(Text.assemble(f"{fmt_ts(event.ts)}  ", (event.kind, "bold"), f"  {json.dumps(event.detail)}"))


@app.command()
def forget(
    item_id: Annotated[str, typer.Argument(help="Fact (f-...) or episode (e-...) id.")],
    user: UserOpt = "default",
    db: DbOpt = Path(DEFAULT_DB),
    embedder: EmbedderOpt = "auto",
) -> None:
    """Delete a fact or episode; a forgotten fact is never learned again from old conversations."""
    if _open(db, user, embedder, offline=True).forget(item_id):
        console.print(f"forgot {item_id} (tombstone kept)")
    else:
        console.print(f"[red]no fact or episode {item_id} in {user}[/red]")
        raise typer.Exit(1)


@app.command()
def bench(
    users: Annotated[int, typer.Option(help="Synthetic users.")] = 50,
    sessions: Annotated[int, typer.Option(help="Sessions per user.")] = 10,
    seed: Annotated[int, typer.Option(help="Generator seed.")] = 7,
    embedder: EmbedderOpt = "auto",
    k: Annotated[int, typer.Option(help="Top-k for recall@k and stale@k.")] = 5,
    budget: Annotated[int, typer.Option(help="Token budget for the packed-context metrics.")] = 200,
    growth_users: Annotated[int, typer.Option(help="Users in the long-horizon store-growth run (0 to skip).")] = 10,
    growth_sessions: Annotated[int, typer.Option(help="Sessions per user in the store-growth run.")] = 40,
    out: Annotated[Path, typer.Option(help="Directory for benchmark.json and benchmark.md.")] = Path(
        "benchmarks/results"
    ),
    charts: Annotated[Path | None, typer.Option(help="Directory for PNG charts (needs matplotlib).")] = Path(
        "assets"
    ),
) -> None:
    """Run the synthetic multi-session benchmark against the baselines and write results (and charts)."""
    from .bench.runner import run_benchmark, write_results

    model = get_embedder(embedder)
    command = (
        f"python -m tiered_memory bench --users {users} --sessions {sessions} --seed {seed} --embedder {embedder} "
        f"--k {k} --budget {budget} --growth-users {growth_users} --growth-sessions {growth_sessions}"
    )
    console.print(f"embedder: {model.name}")
    result = run_benchmark(
        embedder=model,
        n_users=users,
        n_sessions=sessions,
        seed=seed,
        k=k,
        budget_tokens=budget,
        growth_users=growth_users,
        growth_sessions=growth_sessions,
        command=command,
        progress=lambda message: console.print(f"[dim]{message}[/dim]"),
    )
    json_path, md_path = write_results(result, out)
    console.print(f"wrote {json_path} and {md_path}")
    if charts is not None:
        from .bench.charts import save_charts

        for path in save_charts(result, charts):
            console.print(f"wrote {path}")
    console.print(md_path.read_text(encoding="utf-8"))


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8707,
    db: DbOpt = Path(DEFAULT_DB),
    embedder: EmbedderOpt = "auto",
    offline: OfflineOpt = False,
) -> None:
    """Serve the HTTP API (FastAPI) for agents written in any language."""
    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(_open(db, "default", embedder, offline)), host=host, port=port)


def main() -> None:
    app()
