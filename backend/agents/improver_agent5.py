"""Improver Agent 5 — Dedicated entirely to self-improving the Variable Extractor."""

import json
import logging
from backend.agents.claude_client import get_llm
from backend.utils.prompt_loader import CONFIG_PATH, load_prompt
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Improver Agent 5. Your sole responsibility in the 13-agent architecture is to dynamically rewrite and improve the System Prompt of Agent 5 (Variable Extractor).

INSTRUCTIONS:
- You will receive Agent 5's current system prompt and user feedback.
- Completely rewrite the prompt to permanently incorporate the feedback.
- Retain all mandatory rules about variable extraction and JSON structure.
- Output ONLY the new raw string. Do not include markdown blocks or preamble."""

async def execute_improvement(feedback: str) -> str:
    current_prompt = load_prompt("agent5")
    if not current_prompt: return "Error: Could not find prompt for agent5"

    llm = get_llm(max_tokens=2500)
    user_msg = f"CURRENT PROMPT:\n---\n{current_prompt}\n---\nFEEDBACK:\n{feedback}\n\nRewrite Agent 5's prompt now:"
    
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)])
    new_prompt = response.content.strip()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f: data = json.load(f)
    data["agent5"] = new_prompt
    with open(CONFIG_PATH, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)

    logger.info("Improver Agent 5 successfully rewrote Agent 5 prompt.")
    return "Improver Agent 5 has successfully rewritten Agent 5's System Prompt."
