from __future__ import annotations

import os
import tempfile

# Tests use tiny matrices; a single BLAS thread avoids per-thread buffer allocation on small CI machines.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# Keep Hypothesis's example database and caches out of the working tree.
os.environ.setdefault("HYPOTHESIS_STORAGE_DIRECTORY", os.path.join(tempfile.gettempdir(), "tiered-memory-hypothesis"))

from datetime import datetime, timezone  # noqa: E402

import pytest  # noqa: E402

from tiered_memory import HashingEmbedder, ManualClock, MemoryConfig, MemorySystem  # noqa: E402

START = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Never reach an LLM provider from tests, even when a key is configured on the machine."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODELS", raising=False)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))  # keeps a developer's local .env out of reach


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(START)


@pytest.fixture
def memory(clock: ManualClock) -> MemorySystem:
    return MemorySystem(namespace="sara", clock=clock, embedder=HashingEmbedder(), offline=True)


def make_memory(clock: ManualClock, **config: object) -> MemorySystem:
    return MemorySystem(
        namespace="sara",
        clock=clock,
        embedder=HashingEmbedder(),
        offline=True,
        config=MemoryConfig(**config),  # type: ignore[arg-type]
    )


def say(memory: MemorySystem, clock: ManualClock, session: str, *lines: str, role: str = "user") -> None:
    """Add user turns one minute apart."""
    for line in lines:
        clock.advance(minutes=1)
        memory.observe(session, role, line)
