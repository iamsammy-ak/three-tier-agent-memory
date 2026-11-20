"""Benchmark runner: ingest synthetic conversations into every backend, probe, and score.

Metrics (all on probe questions asked after the last session):

* **fact recall@k**: the current value appears in the top-k returned items.
* **stale-fact rate@k**: for facts that changed, an outdated value appears in the top-k items.
* **recall / stale @budget**: the same, for everything packed into a fixed token budget.
* **retrieval latency**: wall time of one top-k recall once the question is embedded (best of three repeats, to
  damp scheduler noise); the time to embed a question is reported separately because it is the same for every
  embedding-based system.
* **store size**: items and kilobytes per user after every session (with and without eviction).

Embeddings are deterministic, so the runner shares one embedding cache across systems: every text is embedded once.
"""

from __future__ import annotations

import json
import platform
import re
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from ..config import MemoryConfig
from ..embeddings import Embedder
from ..extraction import RuleBasedExtractor
from ..predicates import normalise_predicate, normalise_text
from .baselines import Backend, FlatVectorStore, RecencyWindow, TieredBackend
from .synthetic import Dataset, Probe, generate

SYSTEM_NAMES = ("recency-window", "flat-vector", "tiered", "tiered-no-supersede", "tiered-no-eviction")


class CachedEmbedder:
    """Memoises another embedder's vectors by text (benchmark-only: the systems share one cache)."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            for text, vector in zip(missing, self.inner.embed(missing), strict=True):
                self._cache[text] = vector
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._cache[t] for t in texts])


class GrowthPoint(BaseModel):
    session: int
    items: float
    kb: float


class SystemResult(BaseModel):
    name: str
    recall_at_k: float
    stale_at_k: float
    recall_at_budget: float
    stale_at_budget: float
    recall_by_kind: dict[str, float]
    stale_at_k_by_coverage: dict[str, float] = Field(
        default_factory=dict,
        description="Stale@k on changed facts, split by whether every statement used a phrasing the rules cover.",
    )
    latency_ms_p50: float
    latency_ms_p95: float
    items_per_user: float
    kb_per_user: float
    growth: list[GrowthPoint]


class ExtractionResult(BaseModel):
    mentions: int
    recall: float
    recall_covered: float
    recall_uncovered: float
    precision: float
    predicted: int


class ConsolidationStats(BaseModel):
    episodes: int
    raw_tokens: int
    kept_tokens: int
    kept_share: float
    facts_added: int
    facts_superseded: int
    facts_reinforced: int


class BenchmarkResult(BaseModel):
    meta: dict[str, object]
    systems: list[SystemResult]
    extraction: ExtractionResult
    consolidation: ConsolidationStats
    long_growth: dict[str, list[GrowthPoint]] = Field(
        default_factory=dict, description="Store size per user over a longer horizon (separate, smaller run)."
    )
    command: str = Field(description="Exact command that reproduces this result.")


def contains(texts: list[str], value: str) -> bool:
    """Whole-word, case-insensitive match of ``value`` in any of ``texts``."""
    pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.I)
    return any(pattern.search(t) for t in texts)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def make_backends(embedder: Embedder, names: tuple[str, ...] = SYSTEM_NAMES) -> list[Backend]:
    base = MemoryConfig()
    factories: dict[str, Callable[[], Backend]] = {
        "recency-window": lambda: RecencyWindow(),
        "flat-vector": lambda: FlatVectorStore(embedder),
        "tiered": lambda: TieredBackend("tiered", embedder, base),
        "tiered-no-supersede": lambda: TieredBackend(
            "tiered-no-supersede", embedder, base.model_copy(update={"supersede": False})
        ),
        "tiered-no-eviction": lambda: TieredBackend(
            "tiered-no-eviction", embedder, base.model_copy(update={"eviction": False})
        ),
    }
    unknown = set(names) - set(factories)
    if unknown:
        raise ValueError(f"unknown systems: {sorted(unknown)}")
    return [factories[n]() for n in names]


def evaluate_extraction(dataset: Dataset) -> ExtractionResult:
    """Score the rule-based extractor against the generator's ground-truth fact mentions."""
    extractor = RuleBasedExtractor()
    hits = {True: 0, False: 0}
    totals = {True: 0, False: 0}
    predicted = correct_predictions = 0
    for user in dataset.users:
        for session in user.sessions:
            for turn in session.turns:
                if turn.role != "user":
                    continue
                candidates = extractor.extract_sentence(turn.text, turn.ts)
                predicted += len(candidates)
                mentions = [m for m in turn.mentions if m.slot != "event"]

                def match(c_pred: str, c_obj: str, slot: str, value: str) -> bool:
                    pred = normalise_predicate(c_pred)
                    pred_ok = pred.startswith("deadline:") if slot == "deadline" else pred == slot
                    return pred_ok and contains([normalise_text(c_obj)], normalise_text(value))

                for c in candidates:
                    if any(match(c.predicate, c.object, m.slot, m.value) for m in mentions):
                        correct_predictions += 1
                for m in mentions:
                    totals[m.covered] += 1
                    if any(match(c.predicate, c.object, m.slot, m.value) for c in candidates):
                        hits[m.covered] += 1
    n = totals[True] + totals[False]
    return ExtractionResult(
        mentions=n,
        recall=(hits[True] + hits[False]) / n if n else 0.0,
        recall_covered=hits[True] / totals[True] if totals[True] else 0.0,
        recall_uncovered=hits[False] / totals[False] if totals[False] else 0.0,
        precision=correct_predictions / predicted if predicted else 0.0,
        predicted=predicted,
    )


def _ingest(backend: Backend, dataset: Dataset) -> tuple[list[GrowthPoint], float]:
    """Replay every session into the backend; return mean store size per user after each session."""
    sizes: dict[int, list[tuple[int, int]]] = {}
    start = time.perf_counter()
    for user in dataset.users:
        for session in user.sessions:
            for turn in session.turns:
                backend.observe(user.user_id, session.index, turn.role, turn.text, turn.ts)
            backend.end_session(user.user_id, session.index, session.turns[-1].ts + 60)
            sizes.setdefault(session.index, []).append(backend.size(user.user_id))
        backend.finish(user.user_id, user.sessions[-1].turns[-1].ts + 86400)
    growth = [
        GrowthPoint(
            session=idx + 1,
            items=round(statistics.mean(i for i, _ in pairs), 2),
            kb=round(statistics.mean(b for _, b in pairs) / 1024, 2),
        )
        for idx, pairs in sorted(sizes.items())
    ]
    return growth, time.perf_counter() - start


def _run_backend(
    backend: Backend, dataset: Dataset, *, k: int, budget_tokens: int, progress: Callable[[str], None] | None
) -> SystemResult:
    growth, ingest = _ingest(backend, dataset)
    if progress:
        progress(f"{backend.name}: ingested in {ingest:.1f}s, probing")

    probes: list[Probe] = [p for u in dataset.users for p in u.probes]
    latencies: list[float] = []
    rec_k = stale_k = rec_b = stale_b = 0
    by_kind: dict[str, list[int]] = {}
    by_coverage: dict[str, list[int]] = {}
    changed = [p for p in probes if p.stale]
    for probe in probes:
        top = backend.query(probe.user_id, probe.question, k=k, budget_tokens=10**6)  # also caches the question
        timings = []
        for _ in range(3):
            t0 = time.perf_counter()
            backend.query(probe.user_id, probe.question, k=k, budget_tokens=10**6)
            timings.append((time.perf_counter() - t0) * 1000)
        latencies.append(min(timings))
        packed = backend.query(probe.user_id, probe.question, k=None, budget_tokens=budget_tokens)
        hit = contains(top, probe.gold)
        rec_k += hit
        rec_b += contains(packed, probe.gold)
        by_kind.setdefault(probe.kind, []).append(int(hit))
        if probe.stale:
            stale_hit = any(contains(top, s) for s in probe.stale)
            stale_k += stale_hit
            stale_b += any(contains(packed, s) for s in probe.stale)
            key = "all_covered" if probe.all_covered else "some_uncovered"
            by_coverage.setdefault(key, []).append(int(stale_hit))
    final = [backend.size(u.user_id) for u in dataset.users]
    return SystemResult(
        name=backend.name,
        recall_at_k=round(rec_k / len(probes), 4),
        stale_at_k=round(stale_k / len(changed), 4) if changed else 0.0,
        recall_at_budget=round(rec_b / len(probes), 4),
        stale_at_budget=round(stale_b / len(changed), 4) if changed else 0.0,
        recall_by_kind={kind: round(sum(v) / len(v), 4) for kind, v in sorted(by_kind.items())},
        stale_at_k_by_coverage={key: round(sum(v) / len(v), 4) for key, v in sorted(by_coverage.items())},
        latency_ms_p50=round(_percentile(latencies, 0.5), 3),
        latency_ms_p95=round(_percentile(latencies, 0.95), 3),
        items_per_user=round(statistics.mean(i for i, _ in final), 2),
        kb_per_user=round(statistics.mean(b for _, b in final) / 1024, 2),
        growth=growth,
    )


def run_benchmark(
    *,
    embedder: Embedder,
    n_users: int = 50,
    n_sessions: int = 10,
    seed: int = 7,
    k: int = 5,
    budget_tokens: int = 200,
    systems: tuple[str, ...] = SYSTEM_NAMES,
    growth_users: int = 0,
    growth_sessions: int = 40,
    command: str = "",
    progress: Callable[[str], None] | None = None,
) -> BenchmarkResult:
    """Generate the dataset, run every backend and return all metrics.

    With ``growth_users`` > 0, a second, longer dataset (``growth_sessions`` per user, different seed) is replayed
    into the flat store and the tiered store with and without eviction, to show store growth over a longer horizon.
    """
    dataset = generate(n_users=n_users, n_sessions=n_sessions, seed=seed)
    probes = [p for u in dataset.users for p in u.probes]
    question_ms = []
    for probe in probes[:50]:
        t0 = time.perf_counter()
        embedder.embed([probe.question])
        question_ms.append((time.perf_counter() - t0) * 1000)
    embedder = CachedEmbedder(embedder)
    results = []
    consolidation = ConsolidationStats(
        episodes=0, raw_tokens=0, kept_tokens=0, kept_share=0.0, facts_added=0, facts_superseded=0, facts_reinforced=0
    )
    for backend in make_backends(embedder, systems):
        results.append(_run_backend(backend, dataset, k=k, budget_tokens=budget_tokens, progress=progress))
        if isinstance(backend, TieredBackend) and backend.name == "tiered":
            reports = [r for r in backend.reports if r.episode_id]
            raw = sum(r.raw_tokens for r in reports)
            kept = sum(r.kept_tokens for r in reports)
            actions = [f.action for r in backend.reports for f in r.facts]
            consolidation = ConsolidationStats(
                episodes=len(reports),
                raw_tokens=raw,
                kept_tokens=kept,
                kept_share=round(kept / raw, 4) if raw else 0.0,
                facts_added=actions.count("added"),
                facts_superseded=actions.count("superseded"),
                facts_reinforced=actions.count("reinforced") + actions.count("merged"),
            )
    long_growth: dict[str, list[GrowthPoint]] = {}
    if growth_users > 0:
        long_data = generate(n_users=growth_users, n_sessions=growth_sessions, seed=seed + 1)
        for backend in make_backends(embedder, ("flat-vector", "tiered-no-eviction", "tiered")):
            long_growth[backend.name], seconds = _ingest(backend, long_data)
            if progress:
                progress(f"{backend.name}: long-horizon growth run in {seconds:.1f}s")
    meta: dict[str, object] = {
        "seed": seed,
        "users": n_users,
        "sessions_per_user": n_sessions,
        "turns": dataset.n_turns,
        "probes": len(probes),
        "probes_by_kind": {kind: sum(p.kind == kind for p in probes) for kind in ("changed", "stable", "event")},
        "changed_probes_by_coverage": {
            "all_covered": sum(bool(p.stale) and p.all_covered for p in probes),
            "some_uncovered": sum(bool(p.stale) and not p.all_covered for p in probes),
        },
        "k": k,
        "budget_tokens": budget_tokens,
        "embedder": embedder.name,
        "question_embedding_ms_p50": round(statistics.median(question_ms), 3),
        "token_counter": "approx",
        "python": platform.python_version(),
        "data": "synthetic (seeded generator in tiered_memory/bench/synthetic.py)",
        "growth_run": {"users": growth_users, "sessions_per_user": growth_sessions, "seed": seed + 1},
    }
    return BenchmarkResult(
        meta=meta,
        systems=results,
        extraction=evaluate_extraction(dataset),
        consolidation=consolidation,
        long_growth=long_growth,
        command=command,
    )


def to_markdown(result: BenchmarkResult) -> str:
    """Human-readable report of a benchmark result."""
    m = result.meta
    lines = [
        "# Benchmark results",
        "",
        f"Synthetic data: {m['users']} users × {m['sessions_per_user']} sessions ({m['turns']} turns), "
        f"{m['probes']} probe questions {m['probes_by_kind']}, seed {m['seed']}, embedder `{m['embedder']}`.",
        f"Latency is retrieval only (best of 3 per question); embedding a question takes a median "
        f"{m['question_embedding_ms_p50']} ms on top for every system except recency-window.",
        "",
        f"Reproduce: `{result.command}`",
        "",
        f"| System | Recall@{m['k']} | Stale@{m['k']} | Recall@{m['budget_tokens']} tok | "
        f"Stale@{m['budget_tokens']} tok | Recall changed / stable / event | p50 ms | p95 ms | Items/user | KB/user |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for s in result.systems:
        kinds = " / ".join(f"{s.recall_by_kind.get(k, 0):.2f}" for k in ("changed", "stable", "event"))
        lines.append(
            f"| {s.name} | {s.recall_at_k:.3f} | {s.stale_at_k:.3f} | {s.recall_at_budget:.3f} | "
            f"{s.stale_at_budget:.3f} | {kinds} | {s.latency_ms_p50:.2f} | {s.latency_ms_p95:.2f} | "
            f"{s.items_per_user:.1f} | {s.kb_per_user:.1f} |"
        )
    e, c = result.extraction, result.consolidation
    cov = result.meta.get("changed_probes_by_coverage", {})
    assert isinstance(cov, dict)
    lines += [
        "",
        "Stale rates are computed over probes whose fact changed at least once.",
        "",
        f"### Stale@{m['k']} split by extraction coverage",
        "",
        "Changed facts whose every statement used a phrasing the rule-based extractor covers "
        f"({cov.get('all_covered', 0)} probes) versus facts with at least one statement it cannot parse "
        f"({cov.get('some_uncovered', 0)} probes).",
        "",
        "| System | all phrasings covered | some phrasings uncovered |",
        "|---|---:|---:|",
    ]
    for s in result.systems:
        split = s.stale_at_k_by_coverage
        lines.append(f"| {s.name} | {split.get('all_covered', 0):.3f} | {split.get('some_uncovered', 0):.3f} |")
    lines += [
        "",
        "## Rule-based fact extraction against ground truth",
        "",
        f"- Mentions: {e.mentions}; recall {e.recall:.3f} (covered phrasings {e.recall_covered:.3f}, "
        f"uncovered phrasings {e.recall_uncovered:.3f}); precision {e.precision:.3f} over {e.predicted} predictions.",
        "",
        "## Consolidation (tiered system)",
        "",
        f"- Episodes created: {c.episodes}; they keep {c.kept_share:.1%} of the raw tokens "
        f"({c.kept_tokens} of {c.raw_tokens}).",
        f"- Facts added {c.facts_added}, superseded {c.facts_superseded}, reinforced or merged {c.facts_reinforced}.",
        "",
        "## Store growth (mean per user after each session)",
        "",
        "| Session | " + " | ".join(s.name for s in result.systems) + " |",
        "|---:|" + "---:|" * len(result.systems),
    ]
    for i in range(len(result.systems[0].growth)):
        row = " | ".join(f"{s.growth[i].items:.1f} items / {s.growth[i].kb:.1f} KB" for s in result.systems)
        lines.append(f"| {i + 1} | {row} |")
    if result.long_growth:
        g = result.meta["growth_run"]
        assert isinstance(g, dict)
        names = list(result.long_growth)
        lines += [
            "",
            f"## Long-horizon store growth ({g['users']} users × {g['sessions_per_user']} sessions, seed {g['seed']})",
            "",
            "| Session | " + " | ".join(names) + " |",
            "|---:|" + "---:|" * len(names),
        ]
        for i in range(0, len(result.long_growth[names[0]]), 5):
            row = " | ".join(
                f"{result.long_growth[n][i].items:.1f} items / {result.long_growth[n][i].kb:.1f} KB" for n in names
            )
            lines.append(f"| {i + 1} | {row} |")
        last = len(result.long_growth[names[0]])
        row = " | ".join(
            f"{result.long_growth[n][-1].items:.1f} items / {result.long_growth[n][-1].kb:.1f} KB" for n in names
        )
        lines.append(f"| {last} | {row} |")
    return "\n".join(lines) + "\n"


def write_results(result: BenchmarkResult, out_dir: Path, stem: str = "benchmark") -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result.model_dump(), indent=2) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(result), encoding="utf-8")
    return json_path, md_path
