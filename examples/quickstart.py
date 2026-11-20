"""Library quick start: two sessions, a changed fact, and a token-budgeted recall.

Run from the repository root:  python examples/quickstart.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tiered_memory import ManualClock, MemorySystem

clock = ManualClock(1_772_442_000)  # 2 March 2026, 09:00 UTC; omit it to use wall-clock time

with tempfile.TemporaryDirectory() as tmp, MemorySystem.open(
    Path(tmp) / "memory.db", namespace="user:sara", clock=clock
) as memory:
    sid = memory.start_session(goal="Help Sara plan her week")
    memory.observe(sid, "user", "I live in Turin and the quarterly report deadline is 20 March.")
    memory.observe(sid, "assistant", "Noted!")
    memory.end_session(sid)  # working memory -> episode -> facts

    clock.advance(days=9)
    sid = memory.start_session()
    memory.observe(sid, "user", "I just moved to Milan. Actually, the report deadline moved to 27 March.")
    memory.end_session(sid)  # the new facts supersede the old ones

    result = memory.recall("Where does Sara live and when is the report due?", budget_tokens=60)
    for item in result.items:
        print(f"{item.tier:9} {item.score.total:.2f}  {item.text}")
    print()
    print(result.as_context())
    print()
    for fact in memory.history("lives_in"):
        print(fact.text, fact.status, fact.valid_to is None)
