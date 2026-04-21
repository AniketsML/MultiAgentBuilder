"""Agent 0 — State Extractor.

Reads the context document and automatically identifies what conversational
states/steps are needed for the flow. This runs BEFORE the user sees the
state list, giving them AI-suggested states to review and edit.
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.json_parser import extract_json, json_ainvoke_with_retry
from backend.utils.context_chunker import chunk_for_agent

logger = logging.getLogger(__name__)

from backend.utils.prompt_loader import load_prompt

def get_system_prompt() -> str:
    return load_prompt("agent0")


async def extract_states(context_doc: str) -> dict:
    """Analyse a context document and return suggested states."""
    llm = get_llm(max_tokens=4096)

    # Chunk context for large docs — Agent 0 focuses on states & transitions
    chunked_doc = chunk_for_agent(context_doc, "agent0")

    user_prompt = f"""Context document:
---
{chunked_doc}
---

Analyse this document and identify all conversational states needed for this bot.
Respond with ONLY the JSON object, nothing else."""

    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=user_prompt),
    ]

    logger.info("Agent 0: Sending context to LLM for state extraction...")
    parsed = await json_ainvoke_with_retry(llm, messages)
    logger.info(f"Agent 0: Got response successfully.")

    # LLM sometimes ignores the dict wrapper and just returns the array of states
    if isinstance(parsed, list):
        parsed = {"states": parsed, "flow_summary": "Extracted states."}

    # Validate structure
    if not isinstance(parsed, dict) or "states" not in parsed:
        raise ValueError("LLM response missing 'states' key")

    # Filter/clean states
    valid_states = []
    for s in parsed["states"]:
        if isinstance(s, dict) and "state_name" in s:
            valid_states.append(s)
        else:
            logger.warning(f"Agent 0: Dropping invalid state entry: {s}")

    # ── MANDATORY DEFAULT ──────────────────────────────────────────────────────
    # global_instructions is ALWAYS the first state, regardless of what the LLM
    # returned. If it already generated one, move it to position 0. If not, inject.
    gi_names = {"global_instructions", "global_instruction"}
    existing_gi = next((s for s in valid_states if s.get("state_name") in gi_names), None)
    if existing_gi:
        valid_states.remove(existing_gi)
    else:
        existing_gi = {
            "state_name": "global_instructions",
            "description": "System-wide persona, tone, guardrails, and rules that apply to every state in the conversation.",
            "owns": ["persona definition", "tone rules", "guardrails", "escalation rules", "fallback behavior"],
            "transitions_to": "First conversation state (inferred from context)",
            "reason": "Permanent default state — every agent must have global instructions.",
        }
        logger.info("Agent 0: Injected default 'global_instructions' state.")

    parsed["states"] = [existing_gi] + valid_states
    # ──────────────────────────────────────────────────────────────────────────

    logger.info(f"Agent 0: Extracted {len(parsed['states'])} states (incl. global_instructions)")
    return parsed
