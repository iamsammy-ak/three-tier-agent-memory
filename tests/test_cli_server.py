from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tiered_memory import HashingEmbedder, ManualClock, MemorySystem
from tiered_memory.cli import app

runner = CliRunner()


def invoke(*args: str) -> str:
    result = runner.invoke(app, list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


def test_demo_runs_offline_and_shows_every_operation(tmp_path: Path) -> None:
    svg = tmp_path / "demo.svg"
    out = invoke("demo", "--embedder", "hashing", "--svg", str(svg))
    for word in ("working memory full", "session ended", "superseded", "expired by TTL", "recall", "valid from"):
        assert word in out
    assert "The user lives in Milan." in out
    assert svg.read_text(encoding="utf-8").startswith("<svg")


def test_remember_recall_inspect_forget_round_trip(tmp_path: Path) -> None:
    db = str(tmp_path / "m.db")
    common = ["--db", db, "--user", "sara", "--embedder", "hashing"]
    assert "added" in invoke("remember", "I live in Turin and my manager is Paolo Ferri.", *common, "--offline")
    assert "superseded" in invoke("remember", "I just moved to Milan.", *common, "--offline")
    out = invoke("recall", "Where does the user live?", *common, "--offline", "--budget", "50")
    assert "The user lives in Milan." in out and "context block" in out
    data = json.loads(invoke("recall", "Where does the user live?", *common, "--offline", "--json", "--history"))
    assert any(i["provenance"]["status"] == "superseded" for i in data["items"])
    out = invoke("inspect", *common, "--history")
    assert "Turin" in out and "semantic memory" in out and "episodic memory" in out
    fact_id = next(i["id"] for i in data["items"] if i["text"] == "The user lives in Milan.")
    assert "forgot" in invoke("forget", fact_id, *common)
    assert runner.invoke(app, ["forget", fact_id, *common]).exit_code == 1


def test_ingest_and_consolidate_with_a_simulated_date(tmp_path: Path) -> None:
    db = str(tmp_path / "m.db")
    transcript = tmp_path / "t.jsonl"
    rows = [
        {"session": "a", "role": "user", "text": "I live in Turin.", "ts": "2026-03-02T09:00:00"},
        {"session": "a", "role": "user", "text": "The weather is grey today.", "ts": "2026-03-02T09:01:00"},
        {"session": "b", "role": "user", "text": "I just moved to Milan.", "ts": "2026-03-20T10:00:00"},
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    common = ["--db", db, "--embedder", "hashing"]
    assert "2 session(s)" in invoke("ingest", str(transcript), *common, "--offline")
    out = invoke("consolidate", *common, "--offline", "--now", "2027-06-01")
    assert "expired" in out
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"role": "user"}\n', encoding="utf-8")
    assert runner.invoke(app, ["ingest", str(bad), *common]).exit_code == 2


def test_embedder_mismatch_is_reported(tmp_path: Path) -> None:
    db = str(tmp_path / "m.db")
    MemorySystem.open(db, embedder=HashingEmbedder(256), offline=True).store.close()
    result = runner.invoke(app, ["inspect", "--db", db, "--embedder", "hashing"])
    assert result.exit_code == 2 and "embedder" in result.output


def test_http_api(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from tiered_memory.server import create_app

    clock = ManualClock(1_772_442_000)
    client = TestClient(create_app(MemorySystem(embedder=HashingEmbedder(), clock=clock, offline=True)))
    assert client.get("/health").json()["mode"] == "offline"
    sid = client.post("/v1/sara/sessions", json={"goal": "help Sara plan"}).json()["session_id"]
    turn = client.post(f"/v1/sara/sessions/{sid}/turns", json={"role": "user", "text": "I live in Turin."})
    assert turn.status_code == 200
    assert client.post(f"/v1/sara/sessions/{sid}/turns", json={"role": "robot", "text": "x"}).status_code == 422
    report = client.post(f"/v1/sara/sessions/{sid}/end").json()
    assert report["facts"][0]["action"] == "added"
    client.post("/v1/sara/remember", json={"text": "I just moved to Milan."})
    recall = client.post("/v1/sara/recall", json={"query": "Which city does the user live in?", "k": 1}).json()
    assert recall["items"][0]["text"] == "The user lives in Milan."
    assert len(client.get("/v1/sara/facts", params={"history": True}).json()) == 2
    assert client.get("/v1/marco/stats").json()["facts_current"] == 0
    assert client.post("/v1/sara/maintain").status_code == 200
    fact_id = recall["items"][0]["id"]
    assert client.delete(f"/v1/sara/items/{fact_id}").json() == {"forgotten": True}
    assert client.delete(f"/v1/sara/items/{fact_id}").status_code == 404
