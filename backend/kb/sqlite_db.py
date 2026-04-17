"""SQLite async helpers for metadata, run tracking, KB record storage,
agent prompt version history, and Prompt DNA persistence."""

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
    """Create tables if they don't exist, and migrate older schemas."""
    db = await _get_db()
    try:
        # ── Step 1: Create tables with the base schema ──
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

            CREATE TABLE IF NOT EXISTS agent_prompt_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                trigger_feedback TEXT DEFAULT '',
                scores_before TEXT DEFAULT '{}',
                scores_after TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS prompt_dna (
                id TEXT PRIMARY KEY,
                source_prompt_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                use_case TEXT NOT NULL DEFAULT '',
                dna_json TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
        """)
        await db.commit()

        # ── Step 2: Migrate — add missing columns to kb_records ──
        cursor = await db.execute("PRAGMA table_info(kb_records)")
        existing_cols = {row[1] for row in await cursor.fetchall()}

        migrations = [
            ("cases_handled",     "TEXT NOT NULL DEFAULT '[]'"),
            ("case_handling_map", "TEXT NOT NULL DEFAULT '{}'"),
            ("variables_used",    "TEXT NOT NULL DEFAULT '[]'"),
            ("transitions",       "TEXT NOT NULL DEFAULT '{}'"),
        ]
        for col_name, col_def in migrations:
            if col_name not in existing_cols:
                await db.execute(
                    f"ALTER TABLE kb_records ADD COLUMN {col_name} {col_def}"
                )
        await db.commit()

        # ── Step 3: Create indexes (safe now that columns exist) ──
        await db.executescript("""
            CREATE INDEX IF NOT EXISTS idx_kb_domain ON kb_records(context_domain);
            CREATE INDEX IF NOT EXISTS idx_kb_cases ON kb_records(cases_handled);
            CREATE INDEX IF NOT EXISTS idx_prompt_history_agent ON agent_prompt_history(agent_id, version);
            CREATE INDEX IF NOT EXISTS idx_dna_domain ON prompt_dna(domain);
            CREATE INDEX IF NOT EXISTS idx_dna_source ON prompt_dna(source_prompt_id);
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


async def get_recent_runs(limit: int = 20) -> list[dict]:
    """Get recent runs for cross-run trend analysis."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


# ──────────────────── KB records ────────────────────

async def insert_kb_record(record: KBRecord):
    db = await _get_db()
    try:
        await db.execute(
            """INSERT OR REPLACE INTO kb_records
               (id, state_name, prompt, context_domain, state_intent,
                tags, source, approved_by, timestamp, run_id,
                cases_handled, case_handling_map, variables_used, transitions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id, record.state_name, record.prompt,
                record.context_domain, record.state_intent,
                json.dumps(record.tags), record.source,
                record.approved_by, record.timestamp, record.run_id,
                json.dumps(record.cases_handled),
                json.dumps(record.case_handling_map),
                json.dumps(record.variables_used),
                json.dumps(record.transitions),
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
            row_dict["cases_handled"] = json.loads(row_dict.get("cases_handled", "[]"))
            row_dict["case_handling_map"] = json.loads(row_dict.get("case_handling_map", "{}"))
            row_dict["variables_used"] = json.loads(row_dict.get("variables_used", "[]"))
            row_dict["transitions"] = json.loads(row_dict.get("transitions", "{}"))
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
                "SELECT id, prompt, state_intent, tags, timestamp, cases_handled FROM kb_records WHERE context_domain = ?",
                (domain,),
            )
        else:
            cursor = await db.execute(
                "SELECT id, prompt, state_intent, tags, timestamp, cases_handled FROM kb_records"
            )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            row_dict = dict(row)
            try:
                row_dict["tags"] = json.loads(row_dict["tags"])
            except (json.JSONDecodeError, TypeError):
                row_dict["tags"] = []
            try:
                row_dict["cases_handled"] = json.loads(row_dict.get("cases_handled", "[]"))
            except (json.JSONDecodeError, TypeError):
                row_dict["cases_handled"] = []
            results.append(row_dict)
        return results
    finally:
        await db.close()


async def get_last_upsert_timestamp(domain: str) -> str | None:
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


async def query_kb_by_case_category(
    domain: str,
    case_categories: list[str],
    limit: int = 10,
) -> list[dict]:
    """Query KB records that handle specific case categories. Used by KB Case Learner."""
    db = await _get_db()
    try:
        # SQLite JSON: search cases_handled array for matching categories
        conditions = []
        params: list = []
        for cat in case_categories:
            conditions.append("cases_handled LIKE ?")
            params.append(f'%"{cat}"%')

        if not conditions:
            return []

        where = " OR ".join(conditions)
        if domain:
            where = f"context_domain = ? AND ({where})"
            params = [domain] + params

        cursor = await db.execute(
            f"SELECT * FROM kb_records WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        )
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            row_dict = dict(row)
            row_dict["tags"] = json.loads(row_dict["tags"])
            row_dict["cases_handled"] = json.loads(row_dict.get("cases_handled", "[]"))
            row_dict["case_handling_map"] = json.loads(row_dict.get("case_handling_map", "{}"))
            row_dict["variables_used"] = json.loads(row_dict.get("variables_used", "[]"))
            row_dict["transitions"] = json.loads(row_dict.get("transitions", "{}"))
            results.append(row_dict)
        return results
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


# ──────────────────── Agent prompt history ────────────────────

async def save_prompt_version(
    agent_id: str,
    prompt: str,
    timestamp: str,
    trigger_feedback: str = "",
    scores_before: dict | None = None,
) -> int:
    """Save a new version of an agent's prompt. Returns the version number."""
    db = await _get_db()
    try:
        # Get current max version
        cursor = await db.execute(
            "SELECT MAX(version) FROM agent_prompt_history WHERE agent_id = ?",
            (agent_id,),
        )
        row = await cursor.fetchone()
        new_version = (row[0] or 0) + 1

        await db.execute(
            """INSERT INTO agent_prompt_history
               (agent_id, version, prompt, timestamp, trigger_feedback, scores_before)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                agent_id, new_version, prompt, timestamp,
                trigger_feedback, json.dumps(scores_before or {}),
            ),
        )
        await db.commit()
        return new_version
    finally:
        await db.close()


async def update_prompt_scores_after(agent_id: str, version: int, scores_after: dict):
    """Update scores_after for a prompt version (called after the next run)."""
    db = await _get_db()
    try:
        await db.execute(
            "UPDATE agent_prompt_history SET scores_after = ? WHERE agent_id = ? AND version = ?",
            (json.dumps(scores_after), agent_id, version),
        )
        await db.commit()
    finally:
        await db.close()


async def get_prompt_history(agent_id: str, limit: int = 3) -> list[dict]:
    """Get the last N versions of an agent's prompt for improvement memory."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM agent_prompt_history WHERE agent_id = ? ORDER BY version DESC LIMIT ?",
            (agent_id, limit),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            row_dict = dict(row)
            row_dict["scores_before"] = json.loads(row_dict.get("scores_before", "{}"))
            row_dict["scores_after"] = json.loads(row_dict.get("scores_after", "{}"))
            results.append(row_dict)
        return results
    finally:
        await db.close()


async def get_last_good_prompt(agent_id: str) -> dict | None:
    """Find the last prompt version where scores improved (for auto-revert)."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            """SELECT * FROM agent_prompt_history
               WHERE agent_id = ? AND scores_after != '{}'
               ORDER BY version DESC LIMIT 5""",
            (agent_id,),
        )
        rows = await cursor.fetchall()
        for row in rows:
            row_dict = dict(row)
            before = json.loads(row_dict.get("scores_before", "{}"))
            after = json.loads(row_dict.get("scores_after", "{}"))
            # Check if scores improved
            if before and after:
                before_avg = sum(before.values()) / len(before) if before else 0
                after_avg = sum(after.values()) / len(after) if after else 0
                if after_avg >= before_avg:
                    row_dict["scores_before"] = before
                    row_dict["scores_after"] = after
                    return row_dict
        return None
    finally:
        await db.close()


async def get_aggregated_score_trends(limit: int = 20) -> list[dict]:
    """Get aggregated review scores across recent runs for Master Agent trend analysis."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT run_id, result_json, created_at FROM runs WHERE status = 'complete' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        trends = []
        for row in rows:
            row_dict = dict(row)
            if row_dict.get("result_json"):
                try:
                    result = json.loads(row_dict["result_json"])
                    trends.append({
                        "run_id": row_dict["run_id"],
                        "created_at": row_dict["created_at"],
                        "draft_count": len(result.get("drafts", [])),
                        "state_count": len(result.get("states", [])),
                    })
                except json.JSONDecodeError:
                    pass
        return trends
    finally:
        await db.close()


# ──────────────────── Prompt DNA persistence ────────────────────

async def insert_dna_record(dna_dict: dict):
    """Store a PromptDNA record in SQLite."""
    db = await _get_db()
    try:
        source_id = dna_dict.get("source_prompt_id", "")
        doc_id = f"dna_{source_id}"
        await db.execute(
            """INSERT OR REPLACE INTO prompt_dna
               (id, source_prompt_id, domain, use_case, dna_json, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                source_id,
                dna_dict.get("domain", ""),
                dna_dict.get("use_case", ""),
                json.dumps(dna_dict),
                dna_dict.get("timestamp", ""),
            ),
        )
        await db.commit()
    finally:
        await db.close()


async def get_dna_for_domain(domain: str) -> list[dict]:
    """Get all DNA records for a domain."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT dna_json FROM prompt_dna WHERE domain = ? ORDER BY timestamp DESC",
            (domain,),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            try:
                results.append(json.loads(row[0]))
            except (json.JSONDecodeError, TypeError):
                pass
        return results
    finally:
        await db.close()


async def get_dna_by_source(source_prompt_id: str) -> dict | None:
    """Get DNA record for a specific source prompt."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT dna_json FROM prompt_dna WHERE source_prompt_id = ?",
            (source_prompt_id,),
        )
        row = await cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    finally:
        await db.close()


async def get_all_dna_records(limit: int = 50) -> list[dict]:
    """Get all DNA records (cross-domain). Used by Paradigm Mixer."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT dna_json FROM prompt_dna ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            try:
                results.append(json.loads(row[0]))
            except (json.JSONDecodeError, TypeError):
                pass
        return results
    finally:
        await db.close()
