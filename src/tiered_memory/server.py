"""HTTP API over a memory store (FastAPI), so agents in any language can use it.

Run with ``tiered-memory serve`` (default port 8707). Every route is scoped to a namespace (a user or an agent).
"""

from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .memory import MemorySystem
from .models import ConsolidationReport, Fact, MaintenanceReport, ObserveResult, RecallResult, StoreStats


class SessionIn(BaseModel):
    session_id: str | None = None
    goal: str | None = None


class SessionOut(BaseModel):
    session_id: str


class TurnIn(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    text: str = Field(min_length=1)


class RememberIn(BaseModel):
    text: str = Field(min_length=1)


class RecallIn(BaseModel):
    query: str = Field(min_length=1)
    budget_tokens: int = Field(256, gt=0, le=32_000)
    k: int | None = Field(None, gt=0)
    include_history: bool = False
    session_id: str | None = None


class ForgetOut(BaseModel):
    forgotten: bool


def create_app(memory: MemorySystem) -> FastAPI:
    """Build the API around ``memory``; other namespaces share its store and components."""
    app = FastAPI(title="tiered-memory", version=__version__)
    systems: dict[str, MemorySystem] = {memory.namespace: memory}
    lock = threading.Lock()

    def ns(namespace: str) -> MemorySystem:
        with lock:
            if namespace not in systems:
                systems[namespace] = memory.for_namespace(namespace)
            return systems[namespace]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__, "mode": memory.mode, "embedder": memory.embedder.name}

    @app.post("/v1/{namespace}/sessions", response_model=SessionOut)
    def start_session(namespace: str, body: SessionIn) -> SessionOut:
        return SessionOut(session_id=ns(namespace).start_session(body.session_id, goal=body.goal))

    @app.post("/v1/{namespace}/sessions/{session_id}/turns", response_model=ObserveResult)
    def observe(namespace: str, session_id: str, body: TurnIn) -> ObserveResult:
        return ns(namespace).observe(session_id, body.role, body.text)

    @app.post("/v1/{namespace}/sessions/{session_id}/end", response_model=ConsolidationReport)
    def end_session(namespace: str, session_id: str) -> ConsolidationReport:
        return ns(namespace).end_session(session_id)

    @app.post("/v1/{namespace}/remember", response_model=ConsolidationReport)
    def remember(namespace: str, body: RememberIn) -> ConsolidationReport:
        return ns(namespace).remember(body.text)

    @app.post("/v1/{namespace}/recall", response_model=RecallResult)
    def recall(namespace: str, body: RecallIn) -> RecallResult:
        return ns(namespace).recall(
            body.query,
            budget_tokens=body.budget_tokens,
            k=body.k,
            include_history=body.include_history,
            session_id=body.session_id,
        )

    @app.post("/v1/{namespace}/maintain", response_model=MaintenanceReport)
    def maintain(namespace: str) -> MaintenanceReport:
        return ns(namespace).maintain()

    @app.get("/v1/{namespace}/facts", response_model=list[Fact])
    def facts(namespace: str, history: bool = False) -> list[Fact]:
        return ns(namespace).facts(include_history=history)

    @app.get("/v1/{namespace}/stats", response_model=StoreStats)
    def stats(namespace: str) -> StoreStats:
        return ns(namespace).stats()

    @app.delete("/v1/{namespace}/items/{item_id}", response_model=ForgetOut)
    def forget(namespace: str, item_id: str) -> ForgetOut:
        if not ns(namespace).forget(item_id):
            raise HTTPException(status_code=404, detail=f"no fact or episode {item_id!r} in {namespace!r}")
        return ForgetOut(forgotten=True)

    return app
