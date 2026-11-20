"""Three-tier long-term memory for LLM agents: working scratchpad, episodic compression and semantic facts."""

from __future__ import annotations

from .clock import ManualClock, SystemClock
from .config import MemoryConfig, ScoringWeights
from .embeddings import FastEmbedEmbedder, HashingEmbedder, get_embedder
from .memory import MemorySystem
from .models import Episode, Fact, RecallItem, RecallResult
from .store import SQLiteStore

__version__ = "0.1.0"

__all__ = [
    "Episode",
    "Fact",
    "FastEmbedEmbedder",
    "HashingEmbedder",
    "ManualClock",
    "MemoryConfig",
    "MemorySystem",
    "RecallItem",
    "RecallResult",
    "SQLiteStore",
    "ScoringWeights",
    "SystemClock",
    "__version__",
    "get_embedder",
]
