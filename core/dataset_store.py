from __future__ import annotations  # PEP 585 generics (list[X]) on Python 3.8

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "datasets.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_sub TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'collecting',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_images (
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    image_id INTEGER NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, image_id)
);
"""

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
class Dataset:
    id: int
    owner_sub: str
    name: str
    description: str
    status: str
    created_at: str
    total_images: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row, total_images: int = 0) -> "Dataset":
        return cls(id=row["id"], owner_sub=row["owner_sub"], name=row["name"],
                   description=row["description"], status=row["status"],
                   created_at=row["created_at"], total_images=total_images)


def _count_images(conn, dataset_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM dataset_images WHERE dataset_id = ?", (dataset_id,)
    ).fetchone()[0]


def create_dataset(owner_sub: str, name: str, description: str = "") -> Dataset:
    now = datetime.utcnow().isoformat()
    with _lock, _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO datasets (owner_sub, name, description, status, created_at) "
            "VALUES (?,?,?,?,?)",
            (owner_sub, name, description, "collecting", now),
        )
        dataset_id = cursor.lastrowid
    return Dataset(id=dataset_id, owner_sub=owner_sub, name=name, description=description,
                    status="collecting", created_at=now, total_images=0)


def list_datasets(owner_sub: str) -> list[Dataset]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM datasets WHERE owner_sub = ? ORDER BY created_at DESC", (owner_sub,)
        ).fetchall()
        return [Dataset.from_row(r, total_images=_count_images(conn, r["id"])) for r in rows]


def get_dataset(dataset_id: int) -> Optional[Dataset]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if not row:
            return None
        return Dataset.from_row(row, total_images=_count_images(conn, dataset_id))


def add_images(dataset_id: int, image_ids: list[int]) -> int:
    """Link images (already saved via core/image_store.py) to a dataset.
    Returns the dataset's new total image count."""
    now = datetime.utcnow().isoformat()
    with _lock, _connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO dataset_images (dataset_id, image_id, added_at) VALUES (?,?,?)",
            [(dataset_id, image_id, now) for image_id in image_ids],
        )
        total = _count_images(conn, dataset_id)
        if total > 0:
            conn.execute("UPDATE datasets SET status = 'ready' WHERE id = ?", (dataset_id,))
    return total


init_db()
