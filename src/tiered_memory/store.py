"""SQLite persistence for all three tiers, namespaced per user or agent.

Vectors are stored as float32 BLOBs next to their rows and searched by brute-force cosine similarity in NumPy, which
is exact and fast enough for the per-namespace sizes a personal memory reaches (thousands of items). Facts have one
vector each; episodes store one vector per kept sentence (plus one for an abstract), so a long episode can still match
a narrow question. One file holds any number of namespaces; every query is filtered by namespace. Parsed rows and
vectors are cached per namespace and invalidated on every write.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .models import Episode, EpisodeSentence, Fact, MemoryEvent, StoreStats, Tier, Tombstone, WorkingItem

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    namespace TEXT NOT NULL, session_id TEXT NOT NULL, started_at REAL NOT NULL, last_activity REAL NOT NULL,
    ended_at REAL, PRIMARY KEY (namespace, session_id));
CREATE TABLE IF NOT EXISTS working (
    id INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT NOT NULL, session_id TEXT NOT NULL, kind TEXT NOT NULL,
    role TEXT NOT NULL, text TEXT NOT NULL, ts REAL NOT NULL, importance REAL NOT NULL, reasons TEXT NOT NULL,
    tokens INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS ix_working_ns ON working (namespace, session_id);
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY, namespace TEXT NOT NULL, session_id TEXT NOT NULL, reason TEXT NOT NULL,
    start_ts REAL NOT NULL, end_ts REAL NOT NULL, sentences TEXT NOT NULL, abstract TEXT,
    importance REAL NOT NULL, raw_tokens INTEGER NOT NULL, tokens INTEGER NOT NULL, created_at REAL NOT NULL,
    last_accessed REAL NOT NULL, access_count INTEGER NOT NULL, expires_at REAL, n_units INTEGER NOT NULL,
    embedding BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS ix_episodes_ns ON episodes (namespace);
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY, namespace TEXT NOT NULL, subject TEXT NOT NULL, predicate TEXT NOT NULL,
    object TEXT NOT NULL, text TEXT NOT NULL, confidence REAL NOT NULL, importance REAL NOT NULL,
    mentions INTEGER NOT NULL, source_episode TEXT, source_session TEXT, source_text TEXT NOT NULL,
    valid_from REAL NOT NULL, valid_to REAL, status TEXT NOT NULL, superseded_by TEXT, created_at REAL NOT NULL,
    last_accessed REAL NOT NULL, access_count INTEGER NOT NULL, expires_at REAL, embedding BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS ix_facts_ns ON facts (namespace, status);
CREATE INDEX IF NOT EXISTS ix_facts_slot ON facts (namespace, subject, predicate);
CREATE TABLE IF NOT EXISTS tombstones (
    id TEXT NOT NULL, namespace TEXT NOT NULL, tier TEXT NOT NULL, label TEXT NOT NULL, reason TEXT NOT NULL,
    content_hash TEXT NOT NULL, created_at REAL NOT NULL, deleted_at REAL NOT NULL, PRIMARY KEY (namespace, id));
CREATE INDEX IF NOT EXISTS ix_tomb_hash ON tombstones (namespace, content_hash);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT NOT NULL, ts REAL NOT NULL, kind TEXT NOT NULL,
    detail TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_events_ns ON events (namespace, id);
"""

_FACT_COLUMNS = [
    "id", "namespace", "subject", "predicate", "object", "text", "confidence", "importance", "mentions",
    "source_episode", "source_session", "source_text", "valid_from", "valid_to", "status", "superseded_by",
    "created_at", "last_accessed", "access_count", "expires_at",
]  # fmt: skip
_EPISODE_SCALARS = [
    "id", "namespace", "session_id", "reason", "start_ts", "end_ts", "abstract", "importance", "raw_tokens", "tokens",
    "created_at", "last_accessed", "access_count", "expires_at",
]  # fmt: skip


class EmbedderMismatchError(RuntimeError):
    """The store was built with a different embedding model; mixing vector spaces would corrupt search."""


def content_hash(*parts: str) -> str:
    """Stable hash used for tombstones and deterministic ids."""
    return hashlib.sha1("\x1f".join(p.casefold().strip() for p in parts).encode("utf-8")).hexdigest()


def _blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


class SQLiteStore:
    """Thread-safe SQLite store. Use ``":memory:"`` for an ephemeral store.

    Read methods return cached objects shared between callers; treat them as read-only (facts are frozen) and
    change state only through the write methods.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._fact_vectors: dict[str, tuple[list[str], np.ndarray]] = {}
        self._episode_vectors: dict[str, dict[str, np.ndarray]] = {}
        self._facts: dict[str, list[Fact]] = {}
        self._episodes: dict[str, list[Episode]] = {}
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", SCHEMA_VERSION)

    # ------------------------------------------------------------------ basics
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self._conn:
            yield self._conn

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def get_meta(self, key: str) -> str | None:
        rows = self._query("SELECT value FROM meta WHERE key = ?", (key,))
        return str(rows[0]["value"]) if rows else None

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as c:
            c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def bind_embedder(self, name: str, dim: int) -> None:
        """Record the embedder on first use; refuse a different one later."""
        stored = self.get_meta("embedder")
        signature = f"{name}|{dim}"
        if stored is None:
            self.set_meta("embedder", signature)
        elif stored != signature:
            raise EmbedderMismatchError(
                f"this store was built with embedder {stored!r} but {signature!r} was requested; "
                "use the same embedder or a new database file"
            )

    def namespaces(self) -> list[str]:
        rows = self._query(
            "SELECT namespace FROM sessions UNION SELECT namespace FROM episodes UNION SELECT namespace FROM facts "
            "ORDER BY 1"
        )
        return [str(r[0]) for r in rows]

    # ---------------------------------------------------------------- sessions
    def touch_session(self, namespace: str, session_id: str, ts: float) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO sessions (namespace, session_id, started_at, last_activity) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (namespace, session_id) DO UPDATE SET last_activity = MAX(last_activity, excluded.last_activity), "  # noqa: E501
                "ended_at = NULL",
                (namespace, session_id, ts, ts),
            )

    def end_session(self, namespace: str, session_id: str, ts: float) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE sessions SET ended_at = ? WHERE namespace = ? AND session_id = ?", (ts, namespace, session_id)
            )

    def open_sessions(self, namespace: str) -> list[tuple[str, float]]:
        """(session_id, last_activity) of sessions that have not ended."""
        rows = self._query(
            "SELECT session_id, last_activity FROM sessions WHERE namespace = ? AND ended_at IS NULL "
            "ORDER BY started_at",
            (namespace,),
        )
        return [(str(r["session_id"]), float(r["last_activity"])) for r in rows]

    # ----------------------------------------------------------------- working
    def add_working(
        self,
        namespace: str,
        session_id: str,
        *,
        kind: str,
        role: str,
        text: str,
        ts: float,
        importance: float,
        reasons: list[str],
        tokens: int,
    ) -> int:
        with self.transaction() as c:
            cur = c.execute(
                "INSERT INTO working (namespace, session_id, kind, role, text, ts, importance, reasons, tokens) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (namespace, session_id, kind, role, text, ts, importance, json.dumps(reasons), tokens),
            )
            return int(cur.lastrowid or 0)

    def working_items(self, namespace: str, session_id: str | None = None) -> list[WorkingItem]:
        sql, params = "SELECT * FROM working WHERE namespace = ?", [namespace]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        rows = self._query(sql + " ORDER BY ts, id", params)
        return [
            WorkingItem(**{**dict(r), "reasons": json.loads(r["reasons"])})
            for r in rows
        ]

    def delete_working(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        with self.transaction() as c:
            c.executemany("DELETE FROM working WHERE id = ?", [(i,) for i in ids])

    # ---------------------------------------------------------------- episodes
    def _invalidate(self, namespace: str, tier: Tier) -> None:
        if tier == "episodic":
            self._episodes.pop(namespace, None)
            self._episode_vectors.pop(namespace, None)
        else:
            self._facts.pop(namespace, None)
            self._fact_vectors.pop(namespace, None)

    def insert_episode(self, episode: Episode, unit_vectors: np.ndarray) -> None:
        """Insert an episode with one vector per kept sentence (in order) plus one for the abstract, if any."""
        expected = len(episode.sentences) + (1 if episode.abstract else 0)
        if unit_vectors.ndim != 2 or unit_vectors.shape[0] != expected:
            raise ValueError(f"expected {expected} unit vectors, got shape {unit_vectors.shape}")
        data = episode.model_dump()
        values = [data[k] for k in _EPISODE_SCALARS]
        sentences = json.dumps([s.model_dump() for s in episode.sentences])
        with self.transaction() as c:
            c.execute(
                f"INSERT INTO episodes ({', '.join(_EPISODE_SCALARS)}, sentences, n_units, embedding) "
                f"VALUES ({', '.join('?' * (len(_EPISODE_SCALARS) + 3))})",
                [*values, sentences, expected, _blob(unit_vectors)],
            )
        self._invalidate(episode.namespace, "episodic")

    @staticmethod
    def _episode(row: sqlite3.Row) -> Episode:
        data = {k: row[k] for k in _EPISODE_SCALARS}
        data["sentences"] = [EpisodeSentence(**s) for s in json.loads(row["sentences"])]
        return Episode(**data)

    def get_episode(self, episode_id: str) -> Episode | None:
        rows = self._query("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        return self._episode(rows[0]) if rows else None

    def episodes(self, namespace: str) -> list[Episode]:
        if namespace not in self._episodes:
            rows = self._query("SELECT * FROM episodes WHERE namespace = ? ORDER BY start_ts, id", (namespace,))
            self._episodes[namespace] = [self._episode(r) for r in rows]
        return list(self._episodes[namespace])

    def episode_vectors(self, namespace: str) -> dict[str, np.ndarray]:
        """Episode id -> matrix of unit vectors (kept sentences, then the abstract)."""
        if namespace not in self._episode_vectors:
            rows = self._query("SELECT id, n_units, embedding FROM episodes WHERE namespace = ?", (namespace,))
            self._episode_vectors[namespace] = {
                str(r["id"]): np.frombuffer(r["embedding"], dtype=np.float32).reshape(int(r["n_units"]), -1)
                for r in rows
            }
        return self._episode_vectors[namespace]

    def episode_exists(self, episode_id: str) -> bool:
        return bool(self._query("SELECT 1 FROM episodes WHERE id = ?", (episode_id,)))

    # ------------------------------------------------------------------- facts
    def insert_fact(self, fact: Fact, vector: np.ndarray) -> None:
        data = fact.model_dump()
        with self.transaction() as c:
            c.execute(
                f"INSERT INTO facts ({', '.join(_FACT_COLUMNS)}, embedding) "
                f"VALUES ({', '.join('?' * (len(_FACT_COLUMNS) + 1))})",
                [*(data[k] for k in _FACT_COLUMNS), _blob(vector)],
            )
        self._invalidate(fact.namespace, "semantic")

    def update_fact(self, fact_id: str, **fields: Any) -> None:
        unknown = set(fields) - set(_FACT_COLUMNS)
        if unknown:
            raise ValueError(f"unknown fact fields: {sorted(unknown)}")
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self.transaction() as c:
            rows = c.execute("SELECT namespace FROM facts WHERE id = ?", (fact_id,)).fetchall()
            c.execute(f"UPDATE facts SET {assignments} WHERE id = ?", [*fields.values(), fact_id])
        for row in rows:
            self._facts.pop(str(row["namespace"]), None)

    def get_fact(self, fact_id: str) -> Fact | None:
        rows = self._query(f"SELECT {', '.join(_FACT_COLUMNS)} FROM facts WHERE id = ?", (fact_id,))
        return Fact(**dict(rows[0])) if rows else None

    def facts(
        self,
        namespace: str,
        *,
        statuses: Sequence[str] | None = None,
        subject: str | None = None,
        predicate: str | None = None,
    ) -> list[Fact]:
        if namespace not in self._facts:
            rows = self._query(
                f"SELECT {', '.join(_FACT_COLUMNS)} FROM facts WHERE namespace = ? ORDER BY valid_from, id",
                (namespace,),
            )
            self._facts[namespace] = [Fact(**dict(r)) for r in rows]
        return [
            f
            for f in self._facts[namespace]
            if (not statuses or f.status in statuses)
            and (subject is None or f.subject == subject)
            and (predicate is None or f.predicate == predicate)
        ]

    def fact_exists(self, fact_id: str) -> bool:
        return bool(self._query("SELECT 1 FROM facts WHERE id = ?", (fact_id,)))

    # ------------------------------------------------------- access + vectors
    def touch(self, namespace: str, tier: Tier, item_id: str, *, ts: float, expires_at: float | None) -> None:
        """Record an access: bump the counter, refresh last_accessed and the expiry (reinforcement)."""
        table = {"episodic": "episodes", "semantic": "facts"}[tier]
        with self.transaction() as c:
            c.execute(
                f"UPDATE {table} SET last_accessed = MAX(last_accessed, ?), access_count = access_count + 1, "
                "expires_at = ? WHERE id = ? AND namespace = ?",
                (ts, expires_at, item_id, namespace),
            )
        if tier == "episodic":
            self._episodes.pop(namespace, None)
        else:
            self._facts.pop(namespace, None)

    def fact_vectors(self, namespace: str) -> tuple[list[str], np.ndarray]:
        """Ids and vectors of every fact in a namespace (cached until the next insert or delete)."""
        if namespace not in self._fact_vectors:
            rows = self._query("SELECT id, embedding FROM facts WHERE namespace = ? ORDER BY rowid", (namespace,))
            ids = [str(r["id"]) for r in rows]
            matrix = (
                np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                if rows
                else np.zeros((0, 0), dtype=np.float32)
            )
            self._fact_vectors[namespace] = (ids, matrix)
        return self._fact_vectors[namespace]

    # -------------------------------------------------------------- deletion
    def delete_item(self, namespace: str, tier: Tier, item_id: str, *, reason: str, label: str, chash: str,
                    created_at: float, deleted_at: float) -> None:
        """Delete an episode or fact and leave a content-free tombstone in its place."""
        table = {"episodic": "episodes", "semantic": "facts"}[tier]
        with self.transaction() as c:
            c.execute(f"DELETE FROM {table} WHERE id = ?", (item_id,))
            c.execute(
                "INSERT OR REPLACE INTO tombstones (id, namespace, tier, label, reason, content_hash, created_at, "
                "deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (item_id, namespace, tier, label, reason, chash, created_at, deleted_at),
            )
        self._invalidate(namespace, tier)

    def is_tombstoned(self, namespace: str, *, item_id: str | None = None, chash: str | None = None,
                      reason: str | None = None) -> bool:
        sql, params = "SELECT 1 FROM tombstones WHERE namespace = ?", [namespace]
        if item_id is not None:
            sql += " AND id = ?"
            params.append(item_id)
        if chash is not None:
            sql += " AND content_hash = ?"
            params.append(chash)
        if reason is not None:
            sql += " AND reason = ?"
            params.append(reason)
        return bool(self._query(sql + " LIMIT 1", params))

    def tombstones(self, namespace: str) -> list[Tombstone]:
        rows = self._query("SELECT * FROM tombstones WHERE namespace = ? ORDER BY deleted_at, id", (namespace,))
        return [Tombstone(**dict(r)) for r in rows]

    # ------------------------------------------------------------------ events
    def log_event(self, namespace: str, ts: float, kind: str, **detail: Any) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO events (namespace, ts, kind, detail) VALUES (?, ?, ?, ?)",
                (namespace, ts, kind, json.dumps(detail, default=str)),
            )

    def events(self, namespace: str, limit: int = 50) -> list[MemoryEvent]:
        rows = self._query(
            "SELECT ts, kind, detail FROM events WHERE namespace = ? ORDER BY id DESC LIMIT ?", (namespace, limit)
        )
        return [MemoryEvent(ts=r["ts"], kind=r["kind"], detail=json.loads(r["detail"])) for r in reversed(rows)]

    # ------------------------------------------------------------------- stats
    def stats(self, namespace: str) -> StoreStats:
        def scalar(sql: str) -> int:
            return int(self._query(sql, (namespace,))[0][0] or 0)

        bytes_ = (
            scalar("SELECT SUM(LENGTH(text)) FROM working WHERE namespace = ?")
            + scalar("SELECT SUM(LENGTH(sentences) + LENGTH(embedding)) FROM episodes WHERE namespace = ?")
            + scalar(
                "SELECT SUM(LENGTH(text) + LENGTH(source_text) + LENGTH(embedding)) FROM facts WHERE namespace = ?"
            )
        )
        return StoreStats(
            namespace=namespace,
            working_items=scalar("SELECT COUNT(*) FROM working WHERE namespace = ?"),
            episodes=scalar("SELECT COUNT(*) FROM episodes WHERE namespace = ?"),
            facts_current=scalar("SELECT COUNT(*) FROM facts WHERE namespace = ? AND status = 'current'"),
            facts_history=scalar("SELECT COUNT(*) FROM facts WHERE namespace = ? AND status != 'current'"),
            tombstones=scalar("SELECT COUNT(*) FROM tombstones WHERE namespace = ?"),
            open_sessions=scalar("SELECT COUNT(*) FROM sessions WHERE namespace = ? AND ended_at IS NULL"),
            approx_bytes=bytes_,
        )
