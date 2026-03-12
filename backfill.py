"""
backfill.py — Generate embeddings for records that predate Phase 2.

Run this once if you have records in memory.db that were inserted before
embeddings.py was wired up (i.e. during Phase 1). After running, all
records will be searchable via semantic recall.

Usage:
    python backfill.py             # embed all records missing embeddings
    python backfill.py --dry-run   # preview what would be processed
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# Add project root to path so local modules resolve correctly
sys.path.insert(0, str(Path(__file__).parent))

import db
import embeddings


def get_unembedded_records() -> list[dict]:
    """Return all records that have no corresponding row in the embeddings table."""
    with db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT r.*
            FROM records r
            LEFT JOIN embeddings e ON e.record_id = r.id
            WHERE e.record_id IS NULL
            ORDER BY r.id ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def backfill(dry_run: bool = False) -> None:
    db.init_db()

    records = get_unembedded_records()

    if not records:
        print("✅ All records already have embeddings. Nothing to do.")
        return

    print(f"Found {len(records)} record(s) without embeddings.\n")

    if dry_run:
        print("DRY RUN — no embeddings will be written:\n")
        for r in records:
            print(f"  [{r['type'].upper()} #{r['id']}] {r['summary']}")
        return

    # Load the model once before the loop (prints a download message on first use)
    print("Loading embedding model...")
    embeddings._get_model()
    print("Model ready.\n")

    success = 0
    failed = 0

    for r in records:
        try:
            import json
            body = json.loads(r["body"]) if r.get("body") else None
            text = embeddings.build_embed_text(r["summary"], body)
            embeddings.store_embedding(r["id"], text)
            print(f"  ✅ [{r['type'].upper()} #{r['id']}] {r['summary'][:70]}")
            success += 1
        except Exception as e:
            print(f"  ❌ [{r['type'].upper()} #{r['id']}] Failed: {e}")
            failed += 1

    print(f"\nDone. {success} embedded, {failed} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill embeddings for existing records.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which records would be processed without writing anything.",
    )
    args = parser.parse_args()
    backfill(dry_run=args.dry_run)
