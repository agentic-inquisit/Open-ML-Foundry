import json
import os
from datetime import datetime
from typing import Optional

from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

from mlops.embedding_service import VisualEmbeddingService

COLLECTION_NAME = "visual_events"
EMBEDDING_DIM = 512  # matches DEFAULT_CLIP_MODEL (openai/clip-vit-base-patch32) in embedding_service.py
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

_embedder: Optional[VisualEmbeddingService] = None
_collection: Optional[Collection] = None
_connected = False


def _get_embedder() -> VisualEmbeddingService:
    global _embedder
    if _embedder is None:
        _embedder = VisualEmbeddingService()
    return _embedder


def _ensure_connected(timeout: float = 3.0) -> None:
    global _connected
    if not _connected:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=timeout)
        _connected = True


def check_connection(timeout: float = 3.0) -> bool:
    """Bounded connectivity probe for use at app startup. Never raises."""
    try:
        _ensure_connected(timeout=timeout)
        return True
    except Exception as e:
        print(f"Milvus not reachable at {MILVUS_HOST}:{MILVUS_PORT} — event ingestion for /detect will be skipped until it's available ({e})")
        return False


def ensure_collection() -> Collection:
    """Return the visual_events collection, creating it (schema + index) on
    first use if it doesn't exist yet. Cached after the first call so
    per-frame ingestion doesn't re-check schema state on every call."""
    global _collection
    if _collection is not None:
        return _collection

    _ensure_connected()

    if utility.has_collection(COLLECTION_NAME):
        _collection = Collection(COLLECTION_NAME)
        _collection.load()
        return _collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="stream_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=4096),
    ]
    schema = CollectionSchema(fields, description="Embedded /detect frames for semantic search")
    collection = Collection(COLLECTION_NAME, schema)
    collection.create_index(
        field_name="embedding",
        index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
    )
    collection.load()

    _collection = collection
    return _collection


def ingest_frame(frame_bytes: bytes, stream_id: str, detections: list,
                  timestamp: Optional[str] = None) -> bool:
    """Embed a captured frame and upsert it into Milvus.

    Best-effort: returns False (and logs) on any failure instead of
    raising, so a missing/unreachable Milvus server never breaks /detect —
    embedding is a side effect of detection, not a precondition for it.
    """
    try:
        collection = ensure_collection()
        embedding = _get_embedder().get_image_embedding(frame_bytes)
        metadata = json.dumps({"detections": detections})[:4096]

        collection.insert([
            [embedding],
            [timestamp or datetime.utcnow().isoformat()],
            [stream_id],
            [metadata],
        ])
        return True
    except Exception as e:
        print(f"Event ingestion skipped (Milvus unavailable or embedding failed): {e}")
        return False