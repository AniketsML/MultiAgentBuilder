"""Improver Agent 6 — Dedicated to self-improving the Pattern Abstractor."""

import json
import logging
from backend.agents.claude_client import get_llm
from backend.utils.prompt_loader import CONFIG_PATH, load_prompt
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Improver Agent 6. Your sole responsibility in the agent architecture is to dynamically rewrite and improve the System Prompt of Agent 6 (Pattern Abstractor).

INSTRUCTIONS:
- You will receive Agent 6's current system prompt and user feedback.
- Completely rewrite the prompt to permanently incorporate the feedback.
- Retain all mandatory rules about output format (template_skeleton, core_rules, anti_patterns, slot_priority).
- The Pattern Abstractor must always output valid JSON with those 4 keys.
- Output ONLY the new raw string. Do not include markdown blocks or preamble."""

async def execute_improvement(feedback: str) -> str:
    """Read the current prompt, improve it via LLM, and update the config."""
    current_prompt = load_prompt("agent6")
    if not current_prompt: return "Error: Could not find prompt for agent6"

    llm = get_llm(max_tokens=2500)
    user_msg = f"CURRENT PROMPT:\n---\n{current_prompt}\n---\nFEEDBACK:\n{feedback}\n\nRewrite Agent 6's prompt now:"
    
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)])
    new_prompt = response.content.strip()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["agent6"] = new_prompt
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info("Improver Agent 6 successfully rewrote Agent 6 prompt.")
    return "Improver Agent 6 has successfully rewritten Agent 6's System Prompt."
