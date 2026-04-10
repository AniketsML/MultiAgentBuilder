"""API routes for approval: POST /approve and POST /discard.

Updated for case-based pipeline: passes case metadata to KB writer,
regeneration re-runs from Prompt Assembler with user feedback.
"""

import json
from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    ApproveRequest, DiscardRequest,
    ContextSchema, PromptDraft, PipelineState,
)
from backend.kb.kb_writer import write_approved
from backend.kb import sqlite_db
from backend.graph.pipeline import get_compiled_regen_graph

router = APIRouter()


@router.post("/approve")
async def approve_draft(request: ApproveRequest):
    """Approve a single draft prompt (as-is or with edits). Writes to KB with case metadata."""
    run_data = await sqlite_db.get_run(request.run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")

    if run_data["status"] != "complete":
        raise HTTPException(status_code=400, detail="Run is not complete yet")

    result = json.loads(run_data["result_json"])
    context = ContextSchema(**result["context"])

    # Find the matching draft
    target_draft = None
    for d in result["drafts"]:
        if d["state_name"] == request.state_name:
            target_draft = d
            break

    if not target_draft:
        raise HTTPException(
            status_code=404,
            detail=f"Draft for state '{request.state_name}' not found",
        )

    was_edited = request.edited_prompt is not None
    draft = PromptDraft(**target_draft)

    if was_edited:
        draft.edit_content = request.edited_prompt
        draft.status = "edited"
    else:
        draft.status = "approved"

    # Write to KB with case metadata
    kb_id = await write_approved(
        draft=draft,
        context=context,
        state_decompositions=result.get("states", []),
        prioritised_cases=result.get("prioritised_cases", []),
        case_handlers=result.get("case_handlers", []),
        run_id=request.run_id,
        was_edited=was_edited,
    )

    # Update the draft status in the stored result
    for d in result["drafts"]:
        if d["state_name"] == request.state_name:
            d["status"] = "edited" if was_edited else "approved"
            if was_edited:
                d["edit_content"] = request.edited_prompt
            break

    await sqlite_db.complete_run(request.run_id, json.dumps(result))

    return {"ok": True, "kb_id": kb_id}


@router.post("/discard")
async def discard_draft(request: DiscardRequest):
    """Discard a draft and optionally regenerate from Prompt Assembler."""
    run_data = await sqlite_db.get_run(request.run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")

    if run_data["status"] != "complete":
        raise HTTPException(status_code=400, detail="Run is not complete yet")

    result = json.loads(run_data["result_json"])

    # Mark the draft as discarded
    for d in result["drafts"]:
        if d["state_name"] == request.state_name:
            d["status"] = "discarded"
            d["discard_reason"] = request.reason
            break

    new_draft = None

    if request.regenerate:
        # Build state for regeneration — re-runs from Prompt Assembler with feedback
        regen_state: PipelineState = {
            "run_id": request.run_id,
            "context_doc": "",
            "raw_text": "",
            "past_prompts": None,
            "state_names": [],
            "context_schema": result["context"],
            # Case-based fields
            "case_learning_contexts": result.get("case_learning_contexts", []),
            "state_decompositions": result.get("states", []),
            "prioritised_cases": result.get("prioritised_cases", []),
            "case_handlers": result.get("case_handlers", []),
            "extracted_variables": result.get("variables", []),
            # Legacy
            "state_specs": [],
            "current_state_index": 0,
            "drafts": result["drafts"],
            "retrieval_contexts": [],
            "review_notes": None,
            "review_findings": [],
            "critic_scorecards": [],
            "progress": "",
            "error": None,
            "is_cold_start": False,
            "cold_start_domains": [],
            "regen_state_name": request.state_name,
            "regen_reason": request.reason or None,
        }

        regen_graph = get_compiled_regen_graph()
        regen_result = await regen_graph.ainvoke(regen_state)

        # Find the regenerated draft
        for d in regen_result.get("drafts", []):
            if d["state_name"] == request.state_name:
                new_draft = d
                break

        # Update stored result with new draft
        if new_draft:
            result["drafts"] = regen_result["drafts"]

    await sqlite_db.complete_run(request.run_id, json.dumps(result))

    return {"ok": True, "new_draft": new_draft}
