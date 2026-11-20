"""Working memory: a token-bounded scratchpad per session (goals, pinned notes and recent turns).

When a new turn pushes the scratchpad over its budget, the oldest turns are handed to consolidation until the
scratchpad is back under ``working_keep_ratio`` of the budget. Goals and notes are pinned: they never overflow and
are folded into the session's final episode when the session ends.
"""

from __future__ import annotations

from .config import MemoryConfig
from .models import WorkingItem
from .store import SQLiteStore
from .tokens import TokenCounter


class WorkingMemory:
    """Scratchpad operations for one namespace (state lives in the store, so it survives restarts)."""

    def __init__(self, store: SQLiteStore, counter: TokenCounter, config: MemoryConfig, namespace: str) -> None:
        self.store = store
        self.counter = counter
        self.config = config
        self.namespace = namespace

    def add(
        self, session_id: str, *, kind: str, role: str, text: str, ts: float, importance: float, reasons: list[str]
    ) -> int:
        self.store.touch_session(self.namespace, session_id, ts)
        return self.store.add_working(
            self.namespace,
            session_id,
            kind=kind,
            role=role,
            text=text,
            ts=ts,
            importance=importance,
            reasons=reasons,
            tokens=self.counter(text),
        )

    def items(self, session_id: str) -> list[WorkingItem]:
        return self.store.working_items(self.namespace, session_id)

    def overflow(self, session_id: str) -> tuple[list[WorkingItem], int]:
        """Oldest turns to flush (empty while the scratchpad fits its budget) and the tokens that remain after."""
        items = self.items(session_id)
        total = sum(i.tokens for i in items)
        if total <= self.config.working_budget_tokens:
            return [], total
        target = self.config.working_budget_tokens * self.config.working_keep_ratio
        turns = [i for i in items if i.kind == "turn"]
        flushable = turns[: max(0, len(turns) - self.config.working_min_turns)]
        flush: list[WorkingItem] = []
        for item in flushable:
            if total <= target:
                break
            flush.append(item)
            total -= item.tokens
        return flush, total

    def render(self, session_id: str) -> str:
        """The scratchpad as a prompt block: goals and notes first, then the recent turns."""
        items = self.items(session_id)
        lines = [f"Goal: {i.text}" for i in items if i.kind == "goal"]
        lines += [f"Note: {i.text}" for i in items if i.kind == "note"]
        lines += [f"{i.role}: {i.text}" for i in items if i.kind == "turn"]
        return "\n".join(lines)
