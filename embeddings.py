"""
embeddings.py — Embedding generation and semantic vector search.

Uses sentence-transformers (all-MiniLM-L6-v2) to generate embeddings
locally — no API key, no cloud, no cost per query.

Stores vectors as BLOBs in the embeddings table (raw float32 bytes)
and performs cosine similarity search by loading all project embeddings
into memory and ranking them with numpy. This is correct and fast at
the scale of a personal dev memory store (hundreds to low thousands
of records).

Public API:
    store_embedding(record_id, text) -> None
    semantic_search(project, query, limit) -> list[dict]
"""

import struct
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

import db

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Load the embedding model once and cache it for the lifetime of the server.

    First call will download ~80MB if the model isn't cached locally.
    Subsequent calls return the already-loaded model instantly.
    """
    return SentenceTransformer(MODEL_NAME)


def _embed(text: str) -> np.ndarray:
    """Generate a normalized float32 embedding vector for the given text."""
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.astype(np.float32)

# ── Serialization ─────────────────────────────────────────────────────────────

def _vector_to_blob(vector: np.ndarray) -> bytes:
    """Pack a float32 numpy array into raw bytes for SQLite BLOB storage."""
    return struct.pack(f"{len(vector)}f", *vector)


def _blob_to_vector(blob: bytes) -> np.ndarray:
    """Unpack raw bytes from SQLite back into a float32 numpy array."""
    n = len(blob) // 4  # 4 bytes per float32
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)

# ── Storage ───────────────────────────────────────────────────────────────────

def store_embedding(record_id: int, text: str) -> None:
    """
    Generate and store an embedding for a record.

    Called immediately after db.insert_record() so every record
    has a corresponding vector from the moment it's created.

    Args:
        record_id: The integer id of the record in the records table.
        text:      The text to embed (typically summary + body fields joined).
    """
    vector = _embed(text)
    blob = _vector_to_blob(vector)

    with db.get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (record_id, embedding) VALUES (?, ?)",
            (record_id, blob),
        )

# ── Search ────────────────────────────────────────────────────────────────────

def semantic_search(project: str, query: str, limit: int = 8) -> list[dict]:
    """
    Find records semantically similar to a query within a project.

    Strategy:
        1. Embed the query with the same model used at insert time.
        2. Load all (record_id, embedding) pairs for the project.
        3. Compute cosine similarity between query and each record.
           (Vectors are pre-normalized at insert time, so dot product = cosine sim.)
        4. Return the top-k records ranked by similarity.

    This in-memory approach is correct and plenty fast for personal-scale
    stores. If a project ever exceeds ~10k records, swap step 2-4 for a
    FAISS index.

    Args:
        project: Project slug to scope the search.
        query:   Natural language query string.
        limit:   Number of results to return.

    Returns:
        List of record dicts (same shape as db._row_to_dict) ordered
        by descending cosine similarity. Empty list if no embeddings exist.
    """
    project = project.strip().lower()
    query_vector = _embed(query)

    # Load all embeddings for this project in one query
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

    # Unpack embeddings into a matrix: shape (n_records, embedding_dim)
    record_ids = [row["id"] for row in rows]
    matrix = np.stack([_blob_to_vector(row["embedding"]) for row in rows])

    # Cosine similarity: dot product of normalized vectors
    scores = matrix @ query_vector  # shape: (n_records,)

    # Rank and take top-k
    top_indices = np.argsort(scores)[::-1][:limit]
    top_ids = [record_ids[i] for i in top_indices]

    # Fetch full record data for the top hits, preserving rank order
    results = []
    for record_id in top_ids:
        record = db.get_record(record_id)
        if record:
            results.append(record)

    return results


def build_embed_text(summary: str, body: dict | None) -> str:
    """
    Construct the text string to embed for a record.

    Joins summary with all non-null body values so the vector
    captures the full semantic content of the record, not just
    the one-line summary.

    Args:
        summary: The record's summary field.
        body:    The record's parsed body dict (or None).

    Returns:
        A single string suitable for embedding.
    """
    parts = [summary]
    if body and isinstance(body, dict):
        for value in body.values():
            if value and isinstance(value, str):
                parts.append(value)
            elif value and isinstance(value, list):
                parts.append(" ".join(str(v) for v in value))
    return " | ".join(parts)
