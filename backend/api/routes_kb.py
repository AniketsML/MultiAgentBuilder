"""API routes for KB management: GET /kb/list and DELETE /kb/{id}."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import KBListResponse, AutoKBRequest, ManualKBEntry, KBRecord
from backend.kb import sqlite_db
from backend.kb import chroma_client
from backend.kb.retrieval_engine import invalidate_bm25_cache
from backend.agents.agent_dna_analyzer import analyze_and_store as extract_dna

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/kb/list")
async def list_kb(
    domain: str | None = None,
    source: str | None = None,
    page: int = 1,
    limit: int = 20,
):
    """Browse KB entries with optional domain and source filters."""
    records, total = await sqlite_db.list_kb_records(
        domain=domain, source=source, page=page, limit=limit
    )
    return KBListResponse(records=records, total=total)


@router.delete("/kb/{record_id}")
async def delete_kb_entry(record_id: str):
    """Remove an entry from both ChromaDB and SQLite."""
    # Delete from ChromaDB
    try:
        chroma_client.delete_from_kb(record_id)
    except Exception:
        pass  # ChromaDB may not have the record

    # Delete from SQLite
    deleted = await sqlite_db.delete_kb_record(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="KB record not found")

    # Invalidate BM25 cache (we don't know the domain, so clear all)
    invalidate_bm25_cache("__all__")

    return {"ok": True}


@router.post("/kb/add")
async def add_kb_entry(request: ManualKBEntry):
    """Manually add a prompt to the Knowledge Base."""
    from uuid import uuid4
    from datetime import datetime, timezone
    
    record = KBRecord(
        id=str(uuid4()),
        state_name=request.state_name,
        prompt=request.prompt,
        context_domain=request.domain,
        state_intent=request.intent,
        tags=request.tags,
        source="edited",  # Treat manual insertions as edited/human-approved
        approved_by="human_manual",
        timestamp=datetime.now(timezone.utc).isoformat(),
        run_id="manual_insert",
    )
    
    # Save to SQLite
    await sqlite_db.insert_kb_record(record)
    
    # Save to ChromaDB (upsert for idempotency)
    chroma_client.upsert_to_kb(record)

    # Invalidate BM25 cache for this domain
    invalidate_bm25_cache(request.domain)

    # Extract Prompt DNA (fire-and-forget) so this prompt contributes to Paradigm Mixer
    try:
        asyncio.create_task(
            extract_dna(
                prompt_text=request.prompt,
                source_prompt_id=record.id,
                domain=request.domain,
                use_case=request.state_name,
            )
        )
        logger.info(f"[KB Add] DNA extraction queued for manual entry {record.id}")
    except Exception as e:
        logger.warning(f"[KB Add] DNA extraction failed to queue: {e}")
    
    return {"ok": True, "id": record.id}


@router.post("/kb/add-auto")
async def add_kb_entry_auto(request: AutoKBRequest):
    """Uses LLM to automatically generate metadata for a pasted prompt and saves to KB."""
    from uuid import uuid4
    from datetime import datetime, timezone
    from backend.agents.claude_client import get_llm
    from backend.utils.json_parser import extract_json
    from langchain_core.messages import HumanMessage, SystemMessage
    
    llm = get_llm(max_tokens=500)
    
    system_prompt = """You are an expert conversational AI designer. 
Analyze the provided conversational bot prompt and extract the metadata.

CRITICAL RULES:
1. DOMAIN: Classify into a broad industry/use-case (e.g., "pizza_delivery", "healthcare_booking", "debt_collection"). Keep it standard and professional. 
2. STATE NAME: Use standard snake_case to describe the PRIMARY PURPOSE of the prompt (what 70%+ of the prompt focuses on). If the primary focus is a conversational step (e.g., collecting payment, booking), name that step (e.g., "collect_payment") EVEN IF it briefly mentions persona traits. ONLY use "global_instructions" if the prompt is entirely about overarching rules and system-wide behaviors.
3. TAGS: Output 1-3 descriptive tags categorizing the type of prompt (e.g., "persona", "guardrails", "routing", "transactional", "fallback"). DO NOT use the domain name as a tag.
4. INTENT: A short sentence describing what the prompt actually achieves.

EXAMPLE (Global instructions for collections):
Prompt: "You are a debt collector. Be firm but polite. Never yell."
JSON: {"domain": "debt_collection", "state_name": "global_instructions", "intent": "Sets the persona and guardrails for the agent", "tags": ["persona", "guardrails"]}

Output ONLY strict JSON in the following format:
{
  "domain": "The generalized use-case",
  "state_name": "The standardized state name",
  "intent": "What the prompt achieves",
  "tags": ["tag1", "tag2"]
}"""

    user_prompt = f"Prompt to analyze:\n{request.prompt}"
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        parsed = extract_json(response.content)
        if isinstance(parsed, list):
            parsed = {}
        
        record = KBRecord(
            id=str(uuid4()),
            state_name=parsed.get("state_name", "unknown_state"),
            prompt=request.prompt.strip(),
            context_domain=parsed.get("domain", "general"),
            state_intent=parsed.get("intent", "Unknown intent"),
            tags=parsed.get("tags", []),
            source="edited",  # Treat manual insertions as edited/human-approved
            approved_by="human_manual_auto",
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id="manual_insert",
        )
        
        # Save to SQLite
        await sqlite_db.insert_kb_record(record)
        
        # Save to ChromaDB (upsert for idempotency)
        chroma_client.upsert_to_kb(record)

        # Invalidate BM25 cache for this domain
        invalidate_bm25_cache(record.context_domain)

        # Extract Prompt DNA (fire-and-forget) so this prompt contributes to Paradigm Mixer
        try:
            asyncio.create_task(
                extract_dna(
                    prompt_text=request.prompt.strip(),
                    source_prompt_id=record.id,
                    domain=record.context_domain,
                    use_case=record.state_name,
                )
            )
            logger.info(f"[KB Add-Auto] DNA extraction queued for auto entry {record.id}")
        except Exception as dna_err:
            logger.warning(f"[KB Add-Auto] DNA extraction failed to queue: {dna_err}")
        
        return {"ok": True, "record": record.model_dump()}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
