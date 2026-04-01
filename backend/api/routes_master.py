"""API routes for the Master Agent (System Brain) chat interface."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.agents.agent_master import run_master_chat

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
        # Convert Pydantic models to dicts
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        
        response = await run_master_chat(request.message, history_dicts)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
