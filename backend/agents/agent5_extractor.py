"""Agent 5 — Variable Extractor node.

Reads the context document and extracts the required variables
that the conversational bot flow needs to collect or use.
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.json_parser import extract_json
from backend.models.schemas import PipelineState, VariableSchema
from backend.utils.context_chunker import chunk_for_agent

logger = logging.getLogger(__name__)

from backend.utils.prompt_loader import load_prompt

def get_system_prompt() -> str:
    return load_prompt("agent5")


async def extract_variables(state: PipelineState) -> dict:
    """Analyse the context document and planned states to extract required variables."""
    raw_text = state.get("raw_text", "")
    state_specs = state.get("state_specs", [])
    
    # Chunk context for large docs — Agent 5 focuses on data & variables
    chunked_text = chunk_for_agent(raw_text, "agent5")

    llm = get_llm(max_tokens=2048)

    import json
    specs_json = json.dumps(state_specs, indent=2)

    user_prompt = f"""Context document:
---
{chunked_text}
---

Planned States for Flow:
{specs_json}

Analyse the document and the planned states to identify all data variables needed for this bot.
Respond with ONLY the JSON object, nothing else."""

    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=user_prompt),
    ]

    logger.info("Agent 5: Sending context to LLM for variable extraction...")
    response = await llm.ainvoke(messages)
    raw_text = response.content
    logger.info(f"Agent 5: Got response ({len(raw_text)} chars)")

    parsed = extract_json(raw_text)

    # Validate structure
    if "variables" not in parsed:
        logger.warning("LLM response missing 'variables' key. Treating as empty.")
        parsed["variables"] = []

    validated_vars = []
    for v in parsed["variables"]:
        if "name" in v and "description" in v and "type" in v:
            validated_vars.append(VariableSchema(**v).model_dump())
        else:
            logger.warning(f"Skipping malformed variable entry: {v}")

    logger.info(f"Agent 5: Extracted {len(validated_vars)} variables")
    return {
        "extracted_variables": validated_vars,
        "progress": "Variables extracted"
    }
