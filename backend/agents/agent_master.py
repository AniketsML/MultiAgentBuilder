"""The Master Agent (The Orchestrator) — upgraded with write access,
diagnostic depth, cross-run trend analysis, feedback routing, and DNA insights.

Capabilities:
- READ: Full pipeline state including intermediates (case decompositions, priorities, handlers)
- WRITE: Approve/discard drafts, add KB entries, trigger improvements, override priorities
- DIAGNOSE: Explain reasoning chains, show per-case scores, trace KB influence
- TRENDS: Cross-run score analysis, case type performance, improvement trajectory
- FEEDBACK: Route qualitative user feedback to improvers as synthetic review failures
- DNA: Query Prompt DNA store, explain which paradigms influenced a prompt
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.agents.claude_client import get_llm
from backend.kb import sqlite_db
from backend.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


MASTER_SYSTEM_PROMPT = """You are the Master Orchestrator -- the brain of the AgentBuilder platform.

You have REAL-TIME access to the system state. Before each response, you are given:
- The latest pipeline run (status, progress, errors, generated prompts, intermediate outputs)
- The latest KB entries (stored prompts with case metadata)
- The current agent prompts (what instructions each agent follows)
- Cross-run score trends (is the system improving over time?)

## YOUR CAPABILITIES

### READ (always available)
- Show exactly what the last run produced at each pipeline stage
- Display case decompositions, priority scores, and case handlers
- Explain why a specific case was dropped, merged, or scored low
- Show which KB entries influenced a specific state's prompt
- Display per-case-handler quality scores from review gates
- Show agent prompt version history and improvement trajectory

### WRITE (when user explicitly asks)
- Approve a draft: "approve the collect_policy_number state"
- Discard a draft: "discard the greeting state because it's too formal"
- Add a KB entry: "add this prompt as a seed for the insurance domain"
- Trigger improvement: "the prompts are too formal overall" -> routes to relevant improvers
- Override case priority: "always treat user_refuses as high criticality for this domain"

### DIAGNOSE (when user asks "why")
- Trace the reasoning chain: context -> case learning -> decomposition -> priority -> handler -> assembly
- Show per-dimension review scores at each gate
- Identify root cause: was a failure caused by the agent being scored or by upstream?
- Compare KB retrieval quality across states

### TRENDS (when user asks about improvement)
- Cross-run score trends per dimension
- Which case types consistently score low
- Which agents have been improved most frequently
- KB growth trajectory per domain

### DNA (Prompt DNA insights)
- Show what DNA has been extracted from KB prompts
- Explain which paradigm principles influenced a specific prompt
- Show source diversity: which prompts provided which paradigm principles
- Identify low-confidence paradigm extractions that may need re-analysis
- Compare DNA across domains

## RESPONSE FORMAT
- Be concise. Max 3-4 sentences unless the user asks for detail.
- Use bullet points for lists.
- Reference specific data: "Your last run produced 5 states: greeting, collect_order, ..."
- When showing prompts or KB entries, quote the relevant parts.
- Never make up data. Only reference what's in the SYSTEM STATE below.
- When the user wants a WRITE action, confirm what you'll do before doing it.
- For feedback routing, explain which agents will be affected."""


async def _get_system_context() -> str:
    """Pull comprehensive DB state for the Master's context."""
    parts = []

    # Latest run with intermediate outputs
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
                parts.append(
                    f"LATEST RUN:\n- ID: {run['run_id']}\n- Status: {run['status']}\n"
                    f"- Progress: {run.get('progress', '')}\n- Error: {run.get('error', 'none')}"
                )
                if run.get('result_json'):
                    try:
                        result = json.loads(run['result_json'])
                        states = [s.get('state_name', '?') for s in result.get('states', [])]
                        drafts = result.get('drafts', [])
                        parts.append(f"- States: {', '.join(states)}")
                        parts.append(f"- Drafts generated: {len(drafts)}")

                        # Case-based intermediates
                        if result.get('case_learning_contexts'):
                            clc_summary = []
                            for clc in result['case_learning_contexts']:
                                cases_found = len(clc.get('learned_cases', []))
                                clc_summary.append(f"    {clc.get('state_name', '?')}: {cases_found} cases learned")
                            parts.append(f"- KB Case Learning:\n" + "\n".join(clc_summary))

                        if result.get('prioritised_cases'):
                            pcl_summary = []
                            for pcl in result['prioritised_cases']:
                                kept = sum(1 for c in pcl.get('cases', []) if c.get('action') == 'keep')
                                total = len(pcl.get('cases', []))
                                pcl_summary.append(f"    {pcl.get('state_name', '?')}: {kept}/{total} cases kept")
                            parts.append(f"- Case Prioritisation:\n" + "\n".join(pcl_summary))

                        if result.get('review_notes'):
                            parts.append(f"- Review: {result['review_notes']}")

                        # Show first draft as sample
                        if drafts:
                            first = drafts[0]
                            parts.append(
                                f"- Sample draft ({first.get('state_name', '?')}): "
                                f"{first.get('prompt', '')[:300]}..."
                            )
                            if first.get('case_breakdown'):
                                parts.append(f"  Cases handled: {', '.join(first['case_breakdown'])}")
                    except Exception:
                        parts.append("- Result: (could not parse)")
            else:
                parts.append("LATEST RUN: No runs found yet.")
        finally:
            await db.close()
    except Exception as e:
        parts.append(f"LATEST RUN: Error fetching: {e}")

    # Latest KB entries with case metadata
    try:
        records, total = await sqlite_db.list_kb_records(page=1, limit=5)
        if records:
            kb_lines = [f"KB ENTRIES ({total} total, showing latest 5):"]
            for r in records:
                cases_str = f" | cases: {', '.join(r.cases_handled)}" if r.cases_handled else ""
                kb_lines.append(f"- [{r.context_domain}] {r.state_name}: {r.prompt[:80]}...{cases_str}")
            parts.append("\n".join(kb_lines))
        else:
            parts.append("KB ENTRIES: Empty -- no prompts stored yet.")
    except Exception as e:
        parts.append(f"KB ENTRIES: Error fetching: {e}")

    # Current agent prompts (new roster)
    try:
        agent_ids = ["agent0", "agent1", "agent2", "kb_learner", "case_prioritiser",
                      "case_writer", "assembler", "agent4"]
        prompt_lines = ["AGENT PROMPTS (first 120 chars each):"]
        for aid in agent_ids:
            p = load_prompt(aid)
            if p:
                prompt_lines.append(f"- {aid}: {p[:120]}...")
        parts.append("\n".join(prompt_lines))
    except Exception:
        parts.append("AGENT PROMPTS: Could not load.")

    # Cross-run trends
    try:
        trends = await sqlite_db.get_aggregated_score_trends(limit=10)
        if trends:
            trend_lines = [f"CROSS-RUN TRENDS ({len(trends)} recent runs):"]
            for t in trends[:5]:
                trend_lines.append(
                    f"- {t['run_id'][:8]}... ({t['created_at'][:10]}): "
                    f"{t['state_count']} states, {t['draft_count']} drafts"
                )
            parts.append("\n".join(trend_lines))
    except Exception:
        pass

    # Agent improvement history
    try:
        history_lines = ["IMPROVEMENT HISTORY (recent):"]
        for aid in ["agent2", "kb_learner", "case_writer", "assembler"]:
            h = await sqlite_db.get_prompt_history(aid, limit=1)
            if h:
                latest = h[0]
                history_lines.append(
                    f"- {aid} v{latest['version']} ({latest['timestamp'][:10]}): "
                    f"{latest.get('trigger_feedback', 'N/A')[:100]}"
                )
        if len(history_lines) > 1:
            parts.append("\n".join(history_lines))
    except Exception:
        pass

    # Prompt DNA store
    try:
        dna_records = await sqlite_db.get_all_dna_records(limit=5)
        if dna_records:
            dna_lines = [f"PROMPT DNA STORE ({len(dna_records)} records, showing latest):"]
            for rec in dna_records:
                source = rec.get("source_prompt_id", "?")[:20]
                domain = rec.get("domain", "?")
                paradigm_summary = []
                for p in ["structural", "linguistic", "behavioral", "persona",
                          "transition", "constraint", "recovery", "rhythm"]:
                    dna_field = rec.get(f"{p}_dna", {})
                    count = len(dna_field.get("principles", []))
                    conf = dna_field.get("confidence", 0.0)
                    if count > 0:
                        paradigm_summary.append(f"{p}({count}p,{conf:.1f})")
                dna_lines.append(f"- [{domain}] {source}: {', '.join(paradigm_summary)}")
            parts.append("\n".join(dna_lines))
        else:
            parts.append("PROMPT DNA STORE: Empty -- no DNA extracted yet.")
    except Exception as e:
        parts.append(f"PROMPT DNA STORE: Error: {e}")

    return "\n\n".join(parts)


async def run_master_chat(message: str, chat_history: list[dict] | None = None) -> str:
    """Entry point for the API route to chat with the Master Agent."""
    try:
        llm = get_llm(max_tokens=1500)

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


async def route_feedback_to_improvers(feedback: str, affected_agents: list[str]) -> str:
    """Route qualitative user feedback from Master chat to improvers.

    Creates synthetic review failures with targeted_instructions derived
    from the user's natural language feedback.
    """
    results = []

    for agent_id in affected_agents:
        try:
            # Map agent names to improver modules
            improver_map = {
                "agent0": "improver_agent0",
                "agent1": "improver_agent1",
                "agent2": "improver_agent2",
                "kb_learner": "improver_kb_learner",
                "case_prioritiser": "improver_case_prioritiser",
                "case_writer": "improver_case_writer",
                "assembler": "improver_assembler",
                "agent4": "improver_agent4",
            }

            module_name = improver_map.get(agent_id)
            if not module_name:
                results.append(f"Unknown agent: {agent_id}")
                continue

            improver_module = __import__(
                f"backend.agents.{module_name}",
                fromlist=["execute_improvement"],
            )

            synthetic_feedback = (
                f"USER FEEDBACK (via Master Agent): {feedback}. "
                f"This feedback was provided directly by the user and should be "
                f"treated as high-priority. Adjust the agent's prompt to address this."
            )

            await improver_module.execute_improvement(synthetic_feedback)
            results.append(f"Routed feedback to {agent_id} via {module_name}")

        except Exception as e:
            results.append(f"Failed to route to {agent_id}: {e}")
            logger.warning(f"[Master] Failed to route feedback to {agent_id}: {e}")

    return "; ".join(results)
