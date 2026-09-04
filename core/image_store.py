from __future__ import annotations  # PEP 585 generics (list[X], tuple[X]) on Python 3.8

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "images.db"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "uploaded_images"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_sub TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    size_mb REAL NOT NULL,
    width INTEGER,
    height INTEGER,
    uploaded_at TEXT NOT NULL,
    tags TEXT
);
"""

_lock = threading.Lock()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.executescript(_SCHEMA)


@dataclass
class ImageRecord:
    id: int
    owner_sub: str
    filename: str
    file_path: str
    size_mb: float
    width: Optional[int]
    height: Optional[int]
    uploaded_at: str
    tags: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ImageRecord":
        return cls(id=row["id"], owner_sub=row["owner_sub"], filename=row["filename"],
                   file_path=row["file_path"], size_mb=row["size_mb"],
                   width=row["width"], height=row["height"],
                   uploaded_at=row["uploaded_at"], tags=row["tags"])


def _safe_filename(filename: str) -> str:
    # Strip any directory components so a crafted filename (e.g.
    # "../../etc/passwd") can't write outside the per-user directory.
    return Path(filename).name


def _image_dimensions(content: bytes) -> tuple[Optional[int], Optional[int]]:
    try:
        from PIL import Image
        import io
        with Image.open(io.BytesIO(content)) as img:
            return img.width, img.height
    except Exception:
        return None, None


def save_image(owner_sub: str, filename: str, content: bytes,
               tags: Optional[str] = None) -> ImageRecord:
    user_dir = IMAGES_DIR / owner_sub
    user_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(filename)
    dest = user_dir / safe_name
    dest.write_bytes(content)

    size_mb = len(content) / (1024 * 1024)
    width, height = _image_dimensions(content)
    uploaded_at = datetime.utcnow().isoformat()

    with _lock, _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO images (owner_sub, filename, file_path, size_mb, width, height, "
            "uploaded_at, tags) VALUES (?,?,?,?,?,?,?,?)",
            (owner_sub, safe_name, str(dest), size_mb, width, height, uploaded_at, tags),
        )
        image_id = cursor.lastrowid

    return ImageRecord(id=image_id, owner_sub=owner_sub, filename=safe_name,
                        file_path=str(dest), size_mb=size_mb, width=width, height=height,
                        uploaded_at=uploaded_at, tags=tags)


def list_images(owner_sub: str, skip: int = 0, limit: int = 10,
                tags: Optional[str] = None) -> tuple[list[ImageRecord], int]:
    query = "SELECT * FROM images WHERE owner_sub = ?"
    params: list = [owner_sub]
    if tags:
        query += " AND tags LIKE ?"
        params.append(f"%{tags}%")

    with _connect() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM ({query})", params
        ).fetchone()[0]
        rows = conn.execute(
            query + " ORDER BY uploaded_at DESC LIMIT ? OFFSET ?", params + [limit, skip]
        ).fetchall()

    return [ImageRecord.from_row(r) for r in rows], total


def get_image(image_id: int) -> Optional[ImageRecord]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    return ImageRecord.from_row(row) if row else None


def delete_image(image_id: int, owner_sub: str) -> bool:
    """Delete the DB record and the file on disk. Returns False if the
    image doesn't exist or isn't owned by owner_sub (no cross-user delete)."""
    record = get_image(image_id)
    if not record or record.owner_sub != owner_sub:
        return False

    Path(record.file_path).unlink(missing_ok=True)
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
    return True


init_db()
