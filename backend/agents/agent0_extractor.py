"""Agent 0 — State Extractor.

Reads the context document and automatically identifies what conversational
states/steps are needed for the flow. This runs BEFORE the user sees the
state list, giving them AI-suggested states to review and edit.
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.json_parser import extract_json
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
    response = await llm.ainvoke(messages)
    raw_text = response.content
    logger.info(f"Agent 0: Got response ({len(raw_text)} chars)")

    parsed = extract_json(raw_text)

    # Validate structure
    if "states" not in parsed:
        raise ValueError("LLM response missing 'states' key")

    for s in parsed["states"]:
        if "state_name" not in s:
            raise ValueError(f"State missing 'state_name': {s}")

    logger.info(f"Agent 0: Extracted {len(parsed['states'])} states")
    return parsed
