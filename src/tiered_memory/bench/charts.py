"""Benchmark charts (matplotlib, white background, colour-blind-safe categorical order)."""

from __future__ import annotations

from pathlib import Path

from .runner import BenchmarkResult

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax: object) -> None:
    ax.spines[["top", "right"]].set_visible(False)  # type: ignore[attr-defined]
    ax.spines[["left", "bottom"]].set_color(MUTED)  # type: ignore[attr-defined]
    ax.tick_params(colors=MUTED, labelsize=10)  # type: ignore[attr-defined]
    ax.grid(axis="y", color=GRID, linewidth=0.8)  # type: ignore[attr-defined]
    ax.set_axisbelow(True)  # type: ignore[attr-defined]


def save_charts(result: BenchmarkResult, out_dir: Path) -> list[Path]:
    """Write ``bench_recall_stale.png`` and ``bench_store_growth.png``; returns the paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    k = result.meta["k"]

    # 1. fact recall@k versus stale-fact rate@k per system
    names = [s.name for s in result.systems]
    recall = [s.recall_at_k for s in result.systems]
    stale = [s.stale_at_k for s in result.systems]
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor="white")
    width = 0.36
    xs = range(len(names))
    bars_r = ax.bar([x - width / 2 - 0.01 for x in xs], recall, width, color=BLUE,
                    label=f"fact recall@{k} (higher is better)")
    bars_s = ax.bar([x + width / 2 + 0.01 for x in xs], stale, width, color=ORANGE,
                    label=f"stale-fact rate@{k} (lower is better)")
    for bars in (bars_r, bars_s):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(xs), names, fontsize=10, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of probe questions", color=MUTED)
    ax.set_title(
        f"Current facts found vs outdated facts returned (top-{k}, {result.meta['probes']} probes, synthetic data)",
        fontsize=11, color=INK, loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=2)
    _style(ax)
    fig.tight_layout()
    path = out_dir / "bench_recall_stale.png"
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    written.append(path)

    # 2. store growth per user, long horizon if available
    series = result.long_growth or {s.name: s.growth for s in result.systems if s.name != "recency-window"}
    colours = {"flat-vector": ORANGE, "tiered-no-eviction": BLUE, "tiered": AQUA}
    labels = {"flat-vector": "flat vector store (every turn)", "tiered-no-eviction": "tiered, no eviction",
              "tiered": "tiered with TTL + capacity eviction"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="white")
    for ax, field, ylabel in ((axes[0], "items", "stored items per user"), (axes[1], "kb", "KB per user")):
        for name, points in series.items():
            if name not in colours:
                continue
            xs_ = [p.session for p in points]
            ys = [getattr(p, field) for p in points]
            ax.plot(xs_, ys, color=colours[name], linewidth=2, label=labels[name])
            ax.annotate(f"{ys[-1]:.0f}", (xs_[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_xlabel("session", color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)
        ax.set_ylim(bottom=0)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    run = result.meta.get("growth_run") if result.long_growth else None
    scope = (f"{run['users']} users × {run['sessions_per_user']} sessions" if isinstance(run, dict)
             else f"{result.meta['users']} users × {result.meta['sessions_per_user']} sessions")
    fig.suptitle(f"Store size over time ({scope}, synthetic data)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path = out_dir / "bench_store_growth.png"
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    written.append(path)
    return written
