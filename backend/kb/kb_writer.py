"""KB write-back and seeding logic."""

import re
from uuid import uuid4
from datetime import datetime, timezone

from backend.models.schemas import KBRecord, ContextSchema, PromptDraft
from backend.kb import chroma_client
from backend.kb import sqlite_db
from backend.kb.retrieval_engine import invalidate_bm25_cache


async def write_approved(
    draft: PromptDraft,
    context: ContextSchema,
    state_specs: list,
    run_id: str,
    was_edited: bool,
):
    """Write an approved prompt to both ChromaDB and SQLite.
    Uses upsert so re-approvals don't create duplicates.
    Invalidates BM25 cache so next run gets fresh index."""

    # Find the matching state spec for tags and intent
    intent = ""
    tags: list[str] = []
    for spec in state_specs:
        if spec.get("state_name") == draft.state_name:
            intent = spec.get("intent", "")
            tags = spec.get("tags", [])
            break

    record = KBRecord(
        id=str(uuid4()),
        state_name=draft.state_name,
        prompt=draft.edit_content if was_edited else draft.prompt,
        context_domain=context.domain,
        state_intent=intent,
        tags=tags,
        source="edited" if was_edited else "approved",
        approved_by="user",
        timestamp=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
    )

    # Upsert to ChromaDB (idempotent — no duplicates)
    chroma_client.upsert_to_kb(record)

    # Write to SQLite
    await sqlite_db.insert_kb_record(record)

    # Invalidate BM25 cache for this domain so next run rebuilds
    invalidate_bm25_cache(context.domain)

    return record.id


async def seed_kb_if_new(past_prompts_text: str, context: ContextSchema, run_id: str):
    """Seed KB with past prompts if this domain hasn't been seeded yet (spec §10).
    
    Idempotent — running twice with the same domain does not create duplicates.
    """
    if await sqlite_db.domain_is_seeded(context.domain):
        return

    raw_chunks = _split_prompts(past_prompts_text)

    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        # Infer a state name from the chunk (simple heuristic)
        state_name = f"seed_prompt_{i + 1}"
        tags = ["seed"]

        record = KBRecord(
            id=str(uuid4()),
            state_name=state_name,
            prompt=chunk,
            context_domain=context.domain,
            state_intent="seed prompt from user examples",
            tags=tags,
            source="seed",
            approved_by="user",
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
        )

        chroma_client.upsert_to_kb(record)
        await sqlite_db.insert_kb_record(record)

    # Invalidate BM25 cache so retrieval gets fresh seeded data
    invalidate_bm25_cache(context.domain)

    await sqlite_db.mark_domain_seeded(
        context.domain, datetime.now(timezone.utc).isoformat()
    )


def _split_prompts(text: str) -> list[str]:
    """Split a block of past prompts by '---' or double newlines."""
    # Try --- separator first
    if "---" in text:
        return [s.strip() for s in re.split(r"\n---+\n", text) if s.strip()]
    # Fall back to double newlines
    chunks = re.split(r"\n\n+", text)
    return [s.strip() for s in chunks if s.strip()]
