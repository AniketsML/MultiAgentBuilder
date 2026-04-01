"""Improver Agent 1 — Dedicated entirely to self-improving the Context Analyser."""

import json
import logging
from backend.agents.claude_client import get_llm
from backend.utils.prompt_loader import CONFIG_PATH, load_prompt
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Improver Agent 1. Your sole responsibility in the 13-agent architecture is to dynamically rewrite and improve the System Prompt of Agent 1 (Context Analyser).

INSTRUCTIONS:
- You will receive Agent 1's current system prompt and user feedback.
- Completely rewrite the prompt to permanently incorporate the feedback.
- Retain all mandatory rules about JSON structure (e.g., 'domain', 'persona', 'format_rules').
- Output ONLY the new raw string. Do not include markdown blocks or preamble."""

async def execute_improvement(feedback: str) -> str:
    current_prompt = load_prompt("agent1")
    if not current_prompt: return "Error: Could not find prompt for agent1"

    llm = get_llm(max_tokens=2500)
    user_msg = f"CURRENT PROMPT:\n---\n{current_prompt}\n---\nFEEDBACK:\n{feedback}\n\nRewrite Agent 1's prompt now:"
    
    response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)])
    new_prompt = response.content.strip()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f: data = json.load(f)
    data["agent1"] = new_prompt
    with open(CONFIG_PATH, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)

    logger.info("Improver Agent 1 successfully rewrote Agent 1 prompt.")
    return "Improver Agent 1 has successfully rewritten Agent 1's System Prompt."
