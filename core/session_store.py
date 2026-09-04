"""
Session Store — SQLite-backed storage for fine-tuning sessions and their
chat-like event history.

A "session" is one fine-tuning conversation: pick a model (LLM or vision),
train it, test it, repeat — with every step recorded as a timestamped event
so the UI can render it like a chat thread.

Deliberately standalone (stdlib sqlite3, no SQLAlchemy), matching
edge/model_registry.py's approach for model versioning. If an ORM gets
adopted later, the table shapes below are the contract to preserve.
"""

from __future__ import annotations  # PEP 585 generics (list[X]) on Python 3.8

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "sessions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model_type TEXT NOT NULL,        -- 'llm' | 'vision'
    model_name TEXT NOT NULL,        -- e.g. 'Qwen/Qwen3.8-Instruct' or 'resnet50'
    model_format TEXT,               -- 'huggingface' | 'gguf' | 'builtin'
    status TEXT NOT NULL DEFAULT 'idle',  -- idle|training|paused|completed|error
    dataset_path TEXT,
    training_config TEXT,            -- JSON blob
    metrics TEXT,                    -- JSON blob (latest metrics snapshot)
    checkpoint_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,        -- see EVENT_TYPES below
    role TEXT NOT NULL,              -- 'user' | 'assistant' | 'system'
    data TEXT NOT NULL,              -- JSON blob
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_session ON session_events(session_id, timestamp);
"""

# Event types the UI knows how to render as chat bubbles.
EVENT_TYPES = {
    "session_created", "user_message", "train_started", "train_progress",
    "train_completed", "train_failed", "inference_request", "inference_result",
    "checkpoint_saved", "session_note",
}

_lock = threading.Lock()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.executescript(_SCHEMA)


@dataclass
class Session:
    id: str
    name: str
    model_type: str
    model_name: str
    model_format: Optional[str]
    status: str
    dataset_path: Optional[str]
    training_config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Session":
        return cls(
            id=row["id"], name=row["name"], model_type=row["model_type"],
            model_name=row["model_name"], model_format=row["model_format"],
            status=row["status"], dataset_path=row["dataset_path"],
            training_config=json.loads(row["training_config"] or "{}"),
            metrics=json.loads(row["metrics"] or "{}"),
            checkpoint_path=row["checkpoint_path"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )


def create_session(name: str, model_type: str, model_name: str,
                    model_format: str = "huggingface",
                    training_config: Optional[dict] = None) -> Session:
    if model_type not in ("llm", "vision"):
        raise ValueError(f"model_type must be 'llm' or 'vision', got {model_type!r}")

    now = datetime.utcnow().isoformat()
    session_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, name, model_type, model_name, model_format, "
            "status, dataset_path, training_config, metrics, checkpoint_path, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, name, model_type, model_name, model_format, "idle",
             None, json.dumps(training_config or {}), "{}", None, now, now),
        )
    session = get_session(session_id)
    add_event(session_id, "session_created", role="system",
              data={"model_name": model_name, "model_type": model_type})
    return session


def get_session(session_id: str) -> Optional[Session]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return Session.from_row(row) if row else None


def list_sessions(model_type: Optional[str] = None) -> list[Session]:
    query = "SELECT * FROM sessions"
    params: tuple = ()
    if model_type:
        query += " WHERE model_type = ?"
        params = (model_type,)
    query += " ORDER BY updated_at DESC"
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [Session.from_row(r) for r in rows]


def update_session(session_id: str, **fields) -> None:
    if not fields:
        return
    allowed = {"status", "dataset_path", "training_config", "metrics", "checkpoint_path"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    for json_field in ("training_config", "metrics"):
        if json_field in updates and not isinstance(updates[json_field], str):
            updates[json_field] = json.dumps(updates[json_field])
    updates["updated_at"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with _connect() as conn:
        conn.execute(f"UPDATE sessions SET {set_clause} WHERE id = ?",
                     (*updates.values(), session_id))


def delete_session(session_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def add_event(session_id: str, event_type: str, role: str = "system",
              data: Optional[dict] = None) -> dict:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event_type: {event_type!r}")
    if role not in ("user", "assistant", "system"):
        raise ValueError(f"role must be user/assistant/system, got {role!r}")

    event_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    payload = data or {}
    with _connect() as conn:
        conn.execute(
            "INSERT INTO session_events (id, session_id, event_type, role, data, timestamp) "
            "VALUES (?,?,?,?,?,?)",
            (event_id, session_id, event_type, role, json.dumps(payload), now),
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    return {"id": event_id, "session_id": session_id, "event_type": event_type,
            "role": role, "data": payload, "timestamp": now}


def get_history(session_id: str, limit: int = 500) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM session_events WHERE session_id = ? "
            "ORDER BY timestamp ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [
        {"id": r["id"], "event_type": r["event_type"], "role": r["role"],
         "data": json.loads(r["data"]), "timestamp": r["timestamp"]}
        for r in rows
    ]


# Initialize on import so callers don't have to remember to.
init_db()
