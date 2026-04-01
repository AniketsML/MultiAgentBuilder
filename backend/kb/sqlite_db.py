"""SQLite async helpers for metadata, run tracking, and KB record storage."""

import json
import os
import aiosqlite
from backend.config import SQLITE_PATH
from backend.models.schemas import KBRecord

_db_path = SQLITE_PATH


async def _get_db() -> aiosqlite.Connection:
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    db = await aiosqlite.connect(_db_path)
    db.row_factory = aiosqlite.Row
    return db


async def init_db():
    """Create tables if they don't exist."""
    db = await _get_db()
    try:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS kb_records (
                id TEXT PRIMARY KEY,
                state_name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                context_domain TEXT NOT NULL,
                state_intent TEXT NOT NULL,
                tags TEXT NOT NULL,
                source TEXT NOT NULL,
                approved_by TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                run_id TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'running',
                progress TEXT DEFAULT '',
                result_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS seeded_domains (
                domain TEXT PRIMARY KEY,
                seeded_at TEXT NOT NULL
            );
        """)
        await db.commit()
    finally:
        await db.close()


# ──────────────────── Run tracking ────────────────────

async def create_run(run_id: str, created_at: str):
    db = await _get_db()
    try:
        await db.execute(
            "INSERT INTO runs (run_id, status, created_at) VALUES (?, 'running', ?)",
            (run_id, created_at),
        )
        await db.commit()
    finally:
        await db.close()


async def update_run_progress(run_id: str, progress: str):
    db = await _get_db()
    try:
        await db.execute(
            "UPDATE runs SET progress = ? WHERE run_id = ?",
            (progress, run_id),
        )
        await db.commit()
    finally:
        await db.close()


async def complete_run(run_id: str, result_json: str):
    db = await _get_db()
    try:
        await db.execute(
            "UPDATE runs SET status = 'complete', result_json = ? WHERE run_id = ?",
            (result_json, run_id),
        )
        await db.commit()
    finally:
        await db.close()


async def fail_run(run_id: str, error: str):
    db = await _get_db()
    try:
        await db.execute(
            "UPDATE runs SET status = 'error', error = ? WHERE run_id = ?",
            (error, run_id),
        )
        await db.commit()
    finally:
        await db.close()


async def get_run(run_id: str) -> dict | None:
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        await db.close()


# ──────────────────── KB records ────────────────────

async def insert_kb_record(record: KBRecord):
    db = await _get_db()
    try:
        await db.execute(
            """INSERT OR REPLACE INTO kb_records
               (id, state_name, prompt, context_domain, state_intent,
                tags, source, approved_by, timestamp, run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id, record.state_name, record.prompt,
                record.context_domain, record.state_intent,
                json.dumps(record.tags), record.source,
                record.approved_by, record.timestamp, record.run_id,
            ),
        )
        await db.commit()
    finally:
        await db.close()


async def list_kb_records(
    domain: str | None = None,
    source: str | None = None,
    page: int = 1,
    limit: int = 20,
) -> tuple[list[KBRecord], int]:
    db = await _get_db()
    try:
        conditions = []
        params: list = []

        if domain:
            conditions.append("context_domain = ?")
            params.append(domain)
        if source:
            conditions.append("source = ?")
            params.append(source)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Count
        cursor = await db.execute(
            f"SELECT COUNT(*) FROM kb_records WHERE {where_clause}", params
        )
        total = (await cursor.fetchone())[0]

        # Paginate
        offset = (page - 1) * limit
        cursor = await db.execute(
            f"SELECT * FROM kb_records WHERE {where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        )
        rows = await cursor.fetchall()

        records = []
        for row in rows:
            row_dict = dict(row)
            row_dict["tags"] = json.loads(row_dict["tags"])
            records.append(KBRecord(**row_dict))

        return records, total
    finally:
        await db.close()


async def delete_kb_record(record_id: str) -> bool:
    db = await _get_db()
    try:
        cursor = await db.execute("DELETE FROM kb_records WHERE id = ?", (record_id,))
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def get_all_prompts_for_bm25(
    domain: str | None = None,
) -> list[dict]:
    """Return all KB records for BM25 indexing, optionally filtered by domain."""
    db = await _get_db()
    try:
        if domain:
            cursor = await db.execute(
                "SELECT id, prompt, state_intent, tags, timestamp FROM kb_records WHERE context_domain = ?",
                (domain,),
            )
        else:
            cursor = await db.execute(
                "SELECT id, prompt, state_intent, tags, timestamp FROM kb_records"
            )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            row_dict = dict(row)
            try:
                row_dict["tags"] = json.loads(row_dict["tags"])
            except (json.JSONDecodeError, TypeError):
                row_dict["tags"] = []
            results.append(row_dict)
        return results
    finally:
        await db.close()


async def get_last_upsert_timestamp(domain: str) -> str | None:
    """Get the most recent timestamp for a domain's KB records.
    Used by BM25 cache to know if a rebuild is needed."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT MAX(timestamp) FROM kb_records WHERE context_domain = ?",
            (domain,),
        )
        row = await cursor.fetchone()
        return row[0] if row and row[0] else None
    finally:
        await db.close()


# ──────────────────── Domain seeding ────────────────────

async def domain_is_seeded(domain: str) -> bool:
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT 1 FROM seeded_domains WHERE domain = ?", (domain,)
        )
        return (await cursor.fetchone()) is not None
    finally:
        await db.close()


async def mark_domain_seeded(domain: str, seeded_at: str):
    db = await _get_db()
    try:
        await db.execute(
            "INSERT OR IGNORE INTO seeded_domains (domain, seeded_at) VALUES (?, ?)",
            (domain, seeded_at),
        )
        await db.commit()
    finally:
        await db.close()
