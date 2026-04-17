"""KB write-back and seeding logic — updated for case-based pipeline with Prompt DNA.

When a prompt is approved, it now:
1. Stores structured case metadata alongside the prompt text
2. Extracts case strategies for the case_strategies_kb collection
3. Extracts Prompt DNA (8 paradigms) and stores in dna_kb
"""

import re
import asyncio
import logging
from uuid import uuid4
from datetime import datetime, timezone

from backend.models.schemas import KBRecord, ContextSchema, PromptDraft
from backend.kb import chroma_client
from backend.kb import sqlite_db
from backend.kb.retrieval_engine import invalidate_bm25_cache
from backend.agents.agent_dna_analyzer import analyze_and_store as extract_dna

logger = logging.getLogger(__name__)


async def write_approved(
    draft: PromptDraft,
    context: ContextSchema,
    state_decompositions: list,
    prioritised_cases: list,
    case_handlers: list,
    run_id: str,
    was_edited: bool,
):
    """Write an approved prompt to both ChromaDB and SQLite with case metadata.

    Uses upsert so re-approvals don't create duplicates.
    Extracts and stores case strategies for future KB Case Learner use.
    """

    # Find matching data for this state
    intent = ""
    tags: list[str] = []
    cases_handled: list[str] = []
    case_handling_map: dict[str, str] = {}
    variables_used: list[str] = []
    transitions: dict[str, str] = {}

    # From decompositions
    for decomp in state_decompositions:
        if decomp.get("state_name") == draft.state_name:
            intent = decomp.get("intent", "")
            tags = decomp.get("tags", [])
            for v in decomp.get("extracted_variables", []):
                if v.get("name"):
                    variables_used.append(v["name"])
            break

    # From prioritised cases
    for pcl in prioritised_cases:
        if pcl.get("state_name") == draft.state_name:
            for c in pcl.get("cases", []):
                if c.get("action") == "keep":
                    cases_handled.append(c.get("category", ""))
                    if c.get("transition_to"):
                        transitions[c["category"]] = c["transition_to"]
            break

    # From case handlers
    for ch_output in case_handlers:
        if ch_output.get("state_name") == draft.state_name:
            for h in ch_output.get("handlers", []):
                cat = h.get("category", "")
                strategy = h.get("bot_response", "")[:200]
                if cat and strategy:
                    case_handling_map[cat] = strategy
                for v in h.get("variables_used", []):
                    if v not in variables_used:
                        variables_used.append(v)
            break

    # Deduplicate
    cases_handled = list(set(cases_handled))
    variables_used = list(set(variables_used))

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
        cases_handled=cases_handled,
        case_handling_map=case_handling_map,
        variables_used=variables_used,
        transitions=transitions,
    )

    # Upsert to ChromaDB (idempotent)
    chroma_client.upsert_to_kb(record)

    # Write to SQLite
    await sqlite_db.insert_kb_record(record)

    # Store individual case strategies for future KB Case Learner
    for cat, strategy in case_handling_map.items():
        strategy_id = f"strategy_{context.domain}_{draft.state_name}_{cat}"
        strategy_text = (
            f"Domain: {context.domain}\n"
            f"State: {draft.state_name}\n"
            f"Case: {cat}\n"
            f"Strategy: {strategy}\n"
            f"Variables: {', '.join(variables_used)}"
        )
        try:
            chroma_client.upsert_case_strategy(
                strategy_id, strategy_text, context.domain, cat
            )
        except Exception:
            pass  # Non-critical

    # Invalidate BM25 cache
    invalidate_bm25_cache(context.domain)

    # Extract Prompt DNA (fire-and-forget async task)
    prompt_text = draft.edit_content if was_edited else draft.prompt
    try:
        asyncio.create_task(
            extract_dna(
                prompt_text=prompt_text,
                source_prompt_id=record.id,
                domain=context.domain,
                use_case=f"{draft.state_name}",
            )
        )
        logger.info(f"[KB Writer] DNA extraction queued for {record.id}")
    except Exception as e:
        logger.warning(f"[KB Writer] DNA extraction failed to queue: {e}")

    return record.id


async def seed_kb_if_new(past_prompts_text: str, context: ContextSchema, run_id: str):
    """Seed KB with past prompts if this domain hasn't been seeded yet."""
    if await sqlite_db.domain_is_seeded(context.domain):
        return

    raw_chunks = _split_prompts(past_prompts_text)

    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

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
            cases_handled=[],
            case_handling_map={},
            variables_used=[],
            transitions={},
        )

        chroma_client.upsert_to_kb(record)
        await sqlite_db.insert_kb_record(record)

    invalidate_bm25_cache(context.domain)

    await sqlite_db.mark_domain_seeded(
        context.domain, datetime.now(timezone.utc).isoformat()
    )

    # Extract DNA from seed prompts (run in background)
    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if not chunk or len(chunk) < 50:
            continue
        try:
            asyncio.create_task(
                extract_dna(
                    prompt_text=chunk,
                    source_prompt_id=f"seed_{context.domain}_{i}",
                    domain=context.domain,
                    use_case="seed_prompt",
                )
            )
        except Exception:
            pass  # Non-critical: DNA is a bonus from seeds


def _split_prompts(text: str) -> list[str]:
    """Split a block of past prompts by '---' or double newlines."""
    if "---" in text:
        return [s.strip() for s in re.split(r"\n---+\n", text) if s.strip()]
    chunks = re.split(r"\n\n+", text)
    return [s.strip() for s in chunks if s.strip()]
