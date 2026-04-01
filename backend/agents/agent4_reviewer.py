"""Agent 4 — Consistency Reviewer node.

Uses the verbatim system prompt from spec §4.4.
Runs once after all state prompts are written.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from backend.utils.json_parser import extract_json

from backend.agents.claude_client import get_llm
from backend.models.schemas import PipelineState, ReviewResult

from backend.utils.prompt_loader import load_prompt

def get_system_prompt() -> str:
    return load_prompt("agent4")


def _build_user_prompt(context_schema_json: str, all_drafts_json: str) -> str:
    return f"""Context schema:
{context_schema_json}

All draft prompts:
{all_drafts_json}

Review for consistency now."""


async def review_consistency(state: PipelineState) -> dict:
    """LangGraph node: review all drafts for cross-prompt consistency."""
    llm = get_llm(max_tokens=2000)

    context_json = json.dumps(state["context_schema"], indent=2)

    # Format drafts for review
    drafts_for_review = []
    for d in state["drafts"]:
        drafts_for_review.append({
            "state_name": d["state_name"],
            "prompt": d["prompt"],
        })
    drafts_json = json.dumps(drafts_for_review, indent=2)

    user_prompt = _build_user_prompt(context_json, drafts_json)

    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw_text = response.content
    parsed = extract_json(raw_text)
    review = ReviewResult(**parsed)

    return {
        "review_notes": review.overall_note,
        "review_findings": [f.model_dump() for f in review.findings],
        "progress": "Consistency review complete",
    }
