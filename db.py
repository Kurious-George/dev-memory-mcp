"""
db.py — SQLite setup and all database operations for dev-memory-mcp.

Handles schema creation, record CRUD, and raw keyword-based recall.
Embedding-based semantic search is handled in embeddings.py, which
calls insert_record() from here after generating the vector.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "memory.db"

VALID_TYPES = {"decision", "dead_end", "context", "question"}

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT    NOT NULL,
    type        TEXT    NOT NULL,
    summary     TEXT    NOT NULL,
    body        TEXT,               -- JSON blob for type-specific fields
    resolved    INTEGER DEFAULT 0,  -- used for 'question' type
    created_at  TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now'))
);

CREATE TABLE IF NOT EXISTS embeddings (
    record_id   INTEGER PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE,
    embedding   BLOB    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_project ON records(project);
CREATE INDEX IF NOT EXISTS idx_type    ON records(type);
CREATE INDEX IF NOT EXISTS idx_project_type ON records(project, type);
"""

# ── Connection ─────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """
    Open a connection with foreign key enforcement and a row factory
    so rows behave like dicts (row["column_name"]).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables and indexes if they don't already exist."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict, parsing the JSON body field."""
    d = dict(row)
    if d.get("body"):
        try:
            d["body"] = json.loads(d["body"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d

# ── Write operations ──────────────────────────────────────────────────────────

def insert_record(
    project: str,
    record_type: str,
    summary: str,
    body: Optional[dict] = None,
) -> int:
    """
    Insert a new record and return its auto-incremented id.

    Args:
        project:     Slug identifier for the project (e.g. "helix", "free-tier-proxy").
        record_type: One of 'decision', 'dead_end', 'context', 'question'.
        summary:     Short, human-readable description of the record.
        body:        Optional dict of type-specific extra fields.

    Returns:
        The integer id of the newly inserted row.
    """
    if record_type not in VALID_TYPES:
        raise ValueError(f"Invalid record type '{record_type}'. Must be one of {VALID_TYPES}.")

    body_json = json.dumps(body) if body else None

    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO records (project, type, summary, body) VALUES (?, ?, ?, ?)",
            (project.strip().lower(), record_type, summary, body_json),
        )
        return cursor.lastrowid


def resolve_question(record_id: int) -> bool:
    """
    Mark a question record as resolved.

    Returns:
        True if a row was updated, False if the id didn't exist.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE records SET resolved = 1 WHERE id = ? AND type = 'question'",
            (record_id,),
        )
        return cursor.rowcount > 0

# ── Read operations ───────────────────────────────────────────────────────────

def get_record(record_id: int) -> Optional[dict]:
    """Fetch a single record by id, or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM records WHERE id = ?", (record_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_records_by_project(
    project: str,
    record_type: Optional[str] = None,
    include_resolved: bool = True,
    limit: int = 50,
) -> list[dict]:
    """
    Fetch records for a project, optionally filtered by type.

    Args:
        project:          Project slug to filter on.
        record_type:      If provided, only return records of this type.
        include_resolved: If False, omit resolved questions.
        limit:            Max rows to return.

    Returns:
        List of record dicts ordered by created_at descending.
    """
    conditions = ["project = ?"]
    params: list = [project.strip().lower()]

    if record_type:
        conditions.append("type = ?")
        params.append(record_type)

    if not include_resolved:
        conditions.append("(type != 'question' OR resolved = 0)")

    params.append(limit)
    where = " AND ".join(conditions)

    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM records WHERE {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()

    return [_row_to_dict(r) for r in rows]


def keyword_search(project: str, query: str, limit: int = 10) -> list[dict]:
    """
    Basic keyword search across summary and body fields for a project.
    Used as the Phase 1 fallback before semantic embeddings are wired up.

    Args:
        project: Project slug to scope the search.
        query:   Free-text search string.
        limit:   Max results to return.

    Returns:
        List of matching record dicts ordered by created_at descending.
    """
    pattern = f"%{query}%"
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM records
            WHERE project = ?
              AND (summary LIKE ? OR body LIKE ?)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (project.strip().lower(), pattern, pattern, limit),
        ).fetchall()

    return [_row_to_dict(r) for r in rows]


def list_projects() -> list[str]:
    """Return a sorted list of all distinct project slugs in the database."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT project FROM records ORDER BY project"
        ).fetchall()
    return [r["project"] for r in rows]


def get_session_brief_data(project: str) -> dict:
    """
    Aggregate the data Claude needs to construct a session brief:
      - 5 most recent context snapshots
      - All unresolved questions
      - 5 most recent decisions
      - 5 most recent dead ends

    Returns a dict with keys: context, questions, decisions, dead_ends.
    """
    project = project.strip().lower()
    with get_connection() as conn:
        context = conn.execute(
            "SELECT * FROM records WHERE project = ? AND type = 'context' ORDER BY created_at DESC LIMIT 5",
            (project,),
        ).fetchall()

        questions = conn.execute(
            "SELECT * FROM records WHERE project = ? AND type = 'question' AND resolved = 0 ORDER BY created_at DESC",
            (project,),
        ).fetchall()

        decisions = conn.execute(
            "SELECT * FROM records WHERE project = ? AND type = 'decision' ORDER BY created_at DESC LIMIT 5",
            (project,),
        ).fetchall()

        dead_ends = conn.execute(
            "SELECT * FROM records WHERE project = ? AND type = 'dead_end' ORDER BY created_at DESC LIMIT 5",
            (project,),
        ).fetchall()

    return {
        "context":   [_row_to_dict(r) for r in context],
        "questions": [_row_to_dict(r) for r in questions],
        "decisions": [_row_to_dict(r) for r in decisions],
        "dead_ends": [_row_to_dict(r) for r in dead_ends],
    }
