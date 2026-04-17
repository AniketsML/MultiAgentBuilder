"""API routes for the Master Agent — chat, write actions, and feedback routing."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.agents.agent_master import run_master_chat, route_feedback_to_improvers
from backend.models.schemas import MasterFeedbackRequest, MasterActionRequest

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class MasterChatRequest(BaseModel):
    message: str
    chat_history: list[ChatMessage] = []


@router.post("/master/chat")
async def chat_with_master(request: MasterChatRequest):
    """Send a message to the Master Orchestrator Agent."""
    try:
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        response = await run_master_chat(request.message, history_dicts)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/master/feedback")
async def route_feedback(request: MasterFeedbackRequest):
    """Route qualitative user feedback to relevant improver agents.

    This is the human feedback ingestion path that isn't tied to approve/discard.
    User feedback like 'too formal across all states' gets routed to the
    relevant improvers as synthetic review failures.
    """
    if not request.feedback:
        raise HTTPException(status_code=400, detail="Feedback cannot be empty")

    if not request.affected_agents:
        # Default: route to case_writer and assembler
        request.affected_agents = ["case_writer", "assembler"]

    try:
        result = await route_feedback_to_improvers(
            request.feedback, request.affected_agents
        )
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/master/action")
async def master_action(request: MasterActionRequest):
    """Execute a write action from the Master Agent.

    Supported actions:
    - approve: Approve a draft (equivalent to POST /approve)
    - discard: Discard a draft
    - trigger_improvement: Route feedback to improvers
    """
    import json
    from backend.kb import sqlite_db

    if request.action == "trigger_improvement":
        feedback = request.payload.get("feedback", "")
        agents = request.payload.get("agents", ["case_writer", "assembler"])
        result = await route_feedback_to_improvers(feedback, agents)
        return {"ok": True, "result": result}

    elif request.action == "approve":
        from backend.api.routes_approve import approve_draft
        from backend.models.schemas import ApproveRequest
        approve_req = ApproveRequest(
            run_id=request.run_id,
            state_name=request.state_name,
            edited_prompt=request.payload.get("edited_prompt"),
        )
        return await approve_draft(approve_req)

    elif request.action == "discard":
        from backend.api.routes_approve import discard_draft
        from backend.models.schemas import DiscardRequest
        discard_req = DiscardRequest(
            run_id=request.run_id,
            state_name=request.state_name,
            reason=request.payload.get("reason", ""),
            regenerate=request.payload.get("regenerate", False),
        )
        return await discard_draft(discard_req)

    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
