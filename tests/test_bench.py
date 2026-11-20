from __future__ import annotations

from pathlib import Path

import pytest

from tiered_memory.bench.baselines import FlatVectorStore, RecencyWindow
from tiered_memory.bench.runner import CachedEmbedder, contains, evaluate_extraction, run_benchmark, write_results
from tiered_memory.bench.synthetic import generate
from tiered_memory.embeddings import HashingEmbedder


def test_generator_is_deterministic_and_seed_dependent() -> None:
    a, b, c = generate(4, 5, seed=3), generate(4, 5, seed=3), generate(4, 5, seed=4)
    assert a.model_dump() == b.model_dump()
    assert a.model_dump() != c.model_dump()
    with pytest.raises(ValueError):
        generate(0, 5)


def test_generated_timelines_are_consistent() -> None:
    data = generate(6, 8, seed=11)
    for user in data.users:
        times = [t.ts for s in user.sessions for t in s.turns]
        assert times == sorted(times)
        kinds = {p.slot: p.kind for p in user.probes}
        assert kinds["event"] == "event" and kinds["dog"] == "stable"
        for probe in user.probes:
            assert probe.gold not in probe.stale
            stated = [m.value for s in user.sessions for t in s.turns for m in t.mentions if m.slot == probe.slot]
            assert stated and stated[-1] == probe.gold
            assert (probe.kind == "changed") == bool(probe.stale)


def test_contains_is_whole_word_and_case_insensitive() -> None:
    assert contains(["The user lives in MILAN."], "Milan")
    assert not contains(["Milano is nice"], "Milan")
    assert contains(["Due on 27 March"], "27 March") and not contains(["Due on 7 March"], "27 March")


def test_baselines() -> None:
    window = RecencyWindow()
    for i in range(6):
        window.observe("u", 0, "user", f"turn {i}", float(i))
    assert window.query("u", "anything", k=2, budget_tokens=10**6) == ["turn 4", "turn 5"]
    assert window.size("u")[0] == 6
    flat = FlatVectorStore(HashingEmbedder())
    for text in ("I live in Turin.", "The match ended 2-1.", "My dog is called Biscuit."):
        flat.observe("u", 0, "user", text, 0.0)
    assert flat.query("u", "dog", k=1, budget_tokens=10**6) == []  # nothing indexed before the session ends
    flat.end_session("u", 0, 1.0)
    assert flat.query("u", "What is the dog called?", k=1, budget_tokens=10**6) == ["My dog is called Biscuit."]


def test_cached_embedder_matches_the_inner_embedder() -> None:
    inner = HashingEmbedder()
    cached = CachedEmbedder(inner)
    texts = ["a b", "c d", "a b"]
    assert (cached.embed(texts) == inner.embed(texts)).all()
    assert cached.embed([]).shape == (0, inner.dim)


def test_rule_extractor_scores_well_on_covered_phrasings_only() -> None:
    result = evaluate_extraction(generate(10, 6, seed=5))
    assert result.recall_covered == 1.0 and result.recall_uncovered == 0.0 and result.precision == 1.0


def test_small_benchmark_end_to_end(tmp_path: Path) -> None:
    result = run_benchmark(embedder=HashingEmbedder(), n_users=2, n_sessions=3, seed=2, growth_users=1,
                           growth_sessions=4, command="test")
    names = [s.name for s in result.systems]
    assert names == ["recency-window", "flat-vector", "tiered", "tiered-no-supersede", "tiered-no-eviction"]
    for system in result.systems:
        assert 0 <= system.recall_at_k <= 1 and 0 <= system.stale_at_k <= 1 and len(system.growth) == 3
        assert set(system.stale_at_k_by_coverage) <= {"all_covered", "some_uncovered"}
    by_name = {s.name: s for s in result.systems}
    assert by_name["tiered"].items_per_user < by_name["flat-vector"].items_per_user
    assert set(result.long_growth) == {"flat-vector", "tiered-no-eviction", "tiered"}
    assert len(result.long_growth["tiered"]) == 4
    json_path, md_path = write_results(result, tmp_path)
    assert json_path.exists() and "| tiered |" in md_path.read_text(encoding="utf-8")
    pytest.importorskip("matplotlib")
    from tiered_memory.bench.charts import save_charts

    for chart in save_charts(result, tmp_path):
        assert chart.stat().st_size > 1000
