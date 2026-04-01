"""API routes for KB management: GET /kb/list and DELETE /kb/{id}."""

from fastapi import APIRouter, HTTPException

from backend.models.schemas import KBListResponse, AutoKBRequest, ManualKBEntry, KBRecord
from backend.kb import sqlite_db
from backend.kb import chroma_client
from backend.kb.retrieval_engine import invalidate_bm25_cache

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
1. Dynamically classify the "state_name" into a clean, highly standardized snake_case category. DO NOT use overly long, ambiguous, or highly specific names (e.g., use "book_appointment" rather than "ask_user_for_doctor_date"). Keep the states broad enough to be reusable but specific enough to be distinct.
2. Dynamically classify the "domain" (use case) into a clean, broad industry/use-case name (e.g., "pizza_delivery", "healthcare_booking", "debt_collection"). Do not restrict yourself to a predefined list, but keep the naming standard and professional.

Output ONLY strict JSON in the following format:
{
  "domain": "The generalized use-case (e.g., hr_support, pizza_delivery)",
  "state_name": "The standardized state name (e.g., greeting, collect_information)",
  "intent": "A short sentence describing what the prompt is trying to achieve",
  "tags": ["tag1", "tag2"]
}"""

    user_prompt = f"Prompt to analyze:\n{request.prompt}"
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        parsed = extract_json(response.content)
        
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
        
        return {"ok": True, "record": record.model_dump()}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
