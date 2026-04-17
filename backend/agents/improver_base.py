"""Base Improver Agent — shared logic for all improver agents.

Features:
- Versioned prompt history with regression detection
- Improvement memory: last 3 versions + feedback + scores
- Auto-revert if two consecutive improvements make things worse
- Cross-agent causation awareness (receives root_cause info)
"""

import json
import logging
from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.prompt_loader import CONFIG_PATH, load_prompt
from backend.kb import sqlite_db

logger = logging.getLogger(__name__)

IMPROVER_SYSTEM_PROMPT = """You are an Improver Agent in a self-improving multi-agent system.

Your sole responsibility is to dynamically rewrite the System Prompt of your assigned
worker agent to incorporate feedback from the review gate.

INSTRUCTIONS:
1. You will receive:
   - The CURRENT system prompt
   - The FEEDBACK from the review gate (what failed and why)
   - PROMPT HISTORY: the last 3 versions of this prompt, what feedback triggered each
     rewrite, and what scores resulted before and after each change
   - Whether this failure was caused by YOUR agent or by an UPSTREAM agent

2. If UPSTREAM CAUSATION is indicated:
   - Your agent may not be at fault. Focus on making your agent MORE RESILIENT to bad
     upstream input rather than changing its core logic.
   - Add defensive instructions like "if the input lacks X, infer from context" rather
     than fundamentally rewriting the prompt.

3. REGRESSION AWARENESS:
   - Check the prompt history. If the previous improvement ALSO failed (scores_after
     worse than scores_before), DO NOT repeat the same approach.
   - Identify what the previous improvement changed and why it didn't work.
   - If two consecutive improvements both made things worse, the system will auto-revert
     to the last good version. Your job is to try a DIFFERENT approach.

4. PRESERVATION:
   - Retain all mandatory rules (formatting, variable extraction, tone matching)
   - Do not remove instructions that were working well
   - Only modify what the feedback specifically targets

5. Output ONLY the new raw system prompt string. No markdown blocks, no preamble,
   no explanation. Just the new prompt text."""


async def execute_improvement(
    agent_id: str,
    feedback: str,
    is_upstream_cause: bool = False,
    upstream_agent: str = "",
) -> str:
    """Execute an improvement for a specific agent with history awareness.

    Args:
        agent_id: The agent whose prompt to improve (e.g., "agent2", "kb_learner")
        feedback: Targeted feedback from the review gate
        is_upstream_cause: Whether the failure was caused by an upstream agent
        upstream_agent: Which upstream agent caused the issue (if applicable)
    """
    current_prompt = load_prompt(agent_id)
    if not current_prompt:
        return f"Error: Could not find prompt for {agent_id}"

    # Get prompt history for improvement memory
    history = await sqlite_db.get_prompt_history(agent_id, limit=3)

    # Check for regression: did the last improvement make things worse?
    regression_note = ""
    consecutive_failures = 0
    if history:
        for h in history:
            before = h.get("scores_before", {})
            after = h.get("scores_after", {})
            if before and after:
                before_avg = sum(before.values()) / len(before) if before else 0
                after_avg = sum(after.values()) / len(after) if after else 0
                if after_avg < before_avg:
                    consecutive_failures += 1
                else:
                    break

    if consecutive_failures >= 2:
        # Auto-revert to last good version
        last_good = await sqlite_db.get_last_good_prompt(agent_id)
        if last_good:
            regression_note = (
                f"\nWARNING: The last {consecutive_failures} improvements ALL made scores worse. "
                f"Auto-reverting to version {last_good['version']} which had better scores. "
                f"Try a FUNDAMENTALLY DIFFERENT approach this time."
            )
            current_prompt = last_good["prompt"]
        else:
            regression_note = (
                f"\nWARNING: The last {consecutive_failures} improvements ALL made scores worse. "
                f"No good version found to revert to. Try a completely different strategy."
            )
    elif consecutive_failures == 1:
        regression_note = (
            f"\nNOTE: The previous improvement made scores WORSE. "
            f"Previous feedback was: '{history[0].get('trigger_feedback', 'unknown')}'. "
            f"Do NOT repeat the same approach. Try something different."
        )

    # Format history for context
    history_text = ""
    if history:
        history_parts = []
        for h in history[:3]:
            history_parts.append(
                f"Version {h['version']} ({h['timestamp']}):\n"
                f"  Trigger: {h.get('trigger_feedback', 'N/A')[:200]}\n"
                f"  Scores before: {json.dumps(h.get('scores_before', {}))}\n"
                f"  Scores after: {json.dumps(h.get('scores_after', {}))}\n"
                f"  Prompt (first 300 chars): {h['prompt'][:300]}..."
            )
        history_text = "\n\n".join(history_parts)

    # Upstream causation context
    upstream_note = ""
    if is_upstream_cause:
        upstream_note = (
            f"\nUPSTREAM CAUSATION: This failure may be caused by {upstream_agent}, "
            f"not your agent. Make your agent more RESILIENT to bad upstream input. "
            f"Add defensive instructions rather than fundamental rewrites."
        )

    llm = get_llm(max_tokens=2500)

    user_msg = f"""CURRENT PROMPT FOR {agent_id}:
---
{current_prompt}
---

FEEDBACK FROM REVIEW GATE:
{feedback}
{regression_note}{upstream_note}

PROMPT HISTORY (last 3 versions):
{history_text if history_text else "No history available yet."}

Rewrite the prompt now. Output ONLY the new prompt text."""

    response = await llm.ainvoke([
        SystemMessage(content=IMPROVER_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    new_prompt = response.content.strip()

    # Remove any markdown wrapping the LLM might add
    if new_prompt.startswith("```"):
        lines = new_prompt.split("\n")
        new_prompt = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    # Save version to history before overwriting
    timestamp = datetime.now(timezone.utc).isoformat()

    # Extract current scores from the feedback for history tracking
    scores_before = {}
    try:
        # Parse dimension scores from feedback if present
        if "scored" in feedback:
            parts = feedback.split("scored")
            for p in parts[1:]:
                score_str = p.strip().split("/")[0].strip()
                try:
                    scores_before["from_feedback"] = int(score_str)
                except ValueError:
                    pass
    except Exception:
        pass

    await sqlite_db.save_prompt_version(
        agent_id=agent_id,
        prompt=current_prompt,  # Save the OLD prompt
        timestamp=timestamp,
        trigger_feedback=feedback[:500],
        scores_before=scores_before,
    )

    # Write new prompt to config
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data[agent_id] = new_prompt
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"[Improver] Successfully rewrote {agent_id} prompt. "
            f"History version saved. Consecutive failures: {consecutive_failures}"
        )
        return f"Improver has successfully rewritten {agent_id}'s System Prompt."
    except Exception as e:
        logger.error(f"[Improver] Failed to write config for {agent_id}: {e}")
        return f"Error writing config: {e}"
