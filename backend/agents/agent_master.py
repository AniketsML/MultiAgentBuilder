"""The Master Agent (The Orchestrator).

This agent sits on the frontend chat widget. It has real-time access to
the pipeline database (latest runs, KB entries, agent prompts) so it can
give informed, precise answers about the system state.
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.agents.claude_client import get_llm
from backend.kb import sqlite_db
from backend.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


MASTER_SYSTEM_PROMPT = """You are the Master Orchestrator — the brain of the AgentBuilder platform.

You have REAL-TIME access to the system state. Before each response, you are given:
- The latest pipeline run (status, progress, errors, generated prompts)
- The latest KB entries (what prompts are stored)
- The current agent prompts (what instructions each agent follows)

USE THIS DATA to give precise, factual answers. Do NOT say "I don't have access" — you DO.

YOU CAN:
- Tell the user exactly what the last pipeline run produced
- Show them the latest KB entries
- Explain what each agent's prompt says
- Diagnose why a run failed by reading the error
- Suggest specific improvements based on actual outputs

RESPONSE FORMAT:
- Be concise. Max 3-4 sentences unless the user asks for detail.
- Use bullet points for lists.
- Reference specific data: "Your last run produced 5 states: greeting, collect_order, ..."
- When showing prompts or KB entries, quote the relevant parts.
- Never make up data. Only reference what's in the SYSTEM STATE below."""


async def _get_system_context() -> str:
    """Pull latest DB state to inject into the Master's context."""
    parts = []

    # Latest run
    try:
        from backend.kb.sqlite_db import _get_db
        db = await _get_db()
        try:
            cursor = await db.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            if row:
                run = dict(row)
                parts.append(f"LATEST RUN:\n- ID: {run['run_id']}\n- Status: {run['status']}\n- Progress: {run.get('progress', '')}\n- Error: {run.get('error', 'none')}")
                if run.get('result_json'):
                    try:
                        result = json.loads(run['result_json'])
                        # Extract key info
                        states = [s.get('state_name', '?') for s in result.get('states', [])]
                        drafts = result.get('drafts', [])
                        parts.append(f"- States: {', '.join(states)}")
                        parts.append(f"- Drafts generated: {len(drafts)}")
                        if result.get('review_notes'):
                            parts.append(f"- Review: {result['review_notes']}")
                        # Include first draft as sample
                        if drafts:
                            first = drafts[0]
                            parts.append(f"- Sample draft ({first.get('state_name', '?')}): {first.get('prompt', '')[:300]}...")
                    except Exception:
                        parts.append("- Result: (could not parse)")
            else:
                parts.append("LATEST RUN: No runs found yet.")
        finally:
            await db.close()
    except Exception as e:
        parts.append(f"LATEST RUN: Error fetching: {e}")

    # Latest KB entries
    try:
        records, total = await sqlite_db.list_kb_records(page=1, limit=5)
        if records:
            kb_lines = [f"KB ENTRIES ({total} total, showing latest 5):"]
            for r in records:
                kb_lines.append(f"- [{r.context_domain}] {r.state_name}: {r.prompt[:100]}...")
            parts.append("\n".join(kb_lines))
        else:
            parts.append("KB ENTRIES: Empty — no prompts stored yet.")
    except Exception as e:
        parts.append(f"KB ENTRIES: Error fetching: {e}")

    # Current agent prompts (just names/first line)
    try:
        prompts = {}
        for i in range(7):
            p = load_prompt(f"agent{i}")
            prompts[f"agent{i}"] = p[:120] + "..."
        prompt_lines = ["AGENT PROMPTS (first 120 chars each):"]
        for k, v in prompts.items():
            prompt_lines.append(f"- {k}: {v}")
        parts.append("\n".join(prompt_lines))
    except Exception:
        parts.append("AGENT PROMPTS: Could not load.")

    return "\n\n".join(parts)


async def run_master_chat(message: str, chat_history: list[dict] | None = None) -> str:
    """Entry point for the API route to chat with the Master Agent."""
    try:
        llm = get_llm(max_tokens=1000)

        # Pull real-time system context
        system_context = await _get_system_context()

        full_system = f"{MASTER_SYSTEM_PROMPT}\n\n--- SYSTEM STATE ---\n{system_context}\n--- END SYSTEM STATE ---"

        messages = [SystemMessage(content=full_system)]

        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=message))

        logger.info(f"[Master] Received message: {message[:100]}...")

        response = await llm.ainvoke(messages)
        result = response.content

        logger.info(f"[Master] Response: {result[:100]}...")
        return result

    except Exception as e:
        logger.error(f"[Master] Error: {e}", exc_info=True)
        return f"Master Agent error: {str(e)}"
