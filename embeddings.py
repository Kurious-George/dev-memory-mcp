"""
embeddings.py — Embedding generation and semantic vector search.

Uses sentence-transformers (all-MiniLM-L6-v2) to generate embeddings
locally — no API key, no cloud, no cost per query.

Stores vectors as BLOBs in the embeddings table (raw float32 bytes)
and performs cosine similarity search by loading all project embeddings
into memory and ranking them with numpy.

Public API:
    store_embedding(record_id, text) -> None        (sync, for backfill.py)
    store_embedding_async(record_id, text) -> None  (async, for tools.py)
    semantic_search(record_id, query, limit) -> list[dict]  (sync)
    semantic_search_async(...) -> list[dict]        (async, for tools.py)
"""

import asyncio
import struct
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

import db

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"

# Single shared executor — keeps torch off the asyncio event loop entirely.
_executor = ThreadPoolExecutor(max_workers=1)


@lru_cache(maxsize=1)
def _get_model():
    """
    Load the embedding model once and cache it for the lifetime of the server.
    Import is lazy so sentence_transformers and torch only load on first tool
    call, not at server startup. MCP handshake completes instantly.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def _embed(text: str) -> np.ndarray:
    """Generate a normalized float32 embedding vector. Blocking — runs in executor when called from async context."""
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.astype(np.float32)


async def _embed_async(text: str) -> np.ndarray:
    """Non-blocking wrapper around _embed for use inside async tool functions."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _embed, text)

# ── Serialization ─────────────────────────────────────────────────────────────

def _vector_to_blob(vector: np.ndarray) -> bytes:
    """Pack a float32 numpy array into raw bytes for SQLite BLOB storage."""
    return struct.pack(f"{len(vector)}f", *vector)


def _blob_to_vector(blob: bytes) -> np.ndarray:
    """Unpack raw bytes from SQLite back into a float32 numpy array."""
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)

# ── Storage ───────────────────────────────────────────────────────────────────

def store_embedding(record_id: int, text: str) -> None:
    """Synchronous version — used by backfill.py."""
    vector = _embed(text)
    blob = _vector_to_blob(vector)
    with db.get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (record_id, embedding) VALUES (?, ?)",
            (record_id, blob),
        )


async def store_embedding_async(record_id: int, text: str) -> None:
    """Async version — used by tools.py. Runs embedding in thread pool."""
    vector = await _embed_async(text)
    blob = _vector_to_blob(vector)
    with db.get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (record_id, embedding) VALUES (?, ?)",
            (record_id, blob),
        )

# ── Search ────────────────────────────────────────────────────────────────────

def _run_search(project: str, query: str, limit: int) -> list[dict]:
    """Core search logic — synchronous, runs in thread pool when called async."""
    project = project.strip().lower()
    query_vector = _embed(query)

    with db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT r.id, e.embedding
            FROM records r
            JOIN embeddings e ON e.record_id = r.id
            WHERE r.project = ?
            """,
            (project,),
        ).fetchall()

    if not rows:
        return []

    record_ids = [row["id"] for row in rows]
    matrix = np.stack([_blob_to_vector(row["embedding"]) for row in rows])
    scores = matrix @ query_vector
    top_indices = np.argsort(scores)[::-1][:limit]
    top_ids = [record_ids[i] for i in top_indices]

    results = []
    for record_id in top_ids:
        record = db.get_record(record_id)
        if record:
            results.append(record)
    return results


def semantic_search(project: str, query: str, limit: int = 8) -> list[dict]:
    """Synchronous semantic search — used by backfill.py and keyword fallback testing."""
    return _run_search(project, query, limit)


async def semantic_search_async(project: str, query: str, limit: int = 8) -> list[dict]:
    """Async semantic search — used by tools.py. Runs in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_search, project, query, limit)

# ── Text construction ─────────────────────────────────────────────────────────

def build_embed_text(summary: str, body: dict | None) -> str:
    """
    Construct the string to embed for a record.
    Joins summary with all non-null body values so the vector captures
    the full semantic content, not just the one-line summary.
    """
    parts = [summary]
    if body and isinstance(body, dict):
        for value in body.values():
            if value and isinstance(value, str):
                parts.append(value)
            elif value and isinstance(value, list):
                parts.append(" ".join(str(v) for v in value))
    return " | ".join(parts)