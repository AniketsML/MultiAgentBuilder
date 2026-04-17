"""Prompt Assembler — new agent.

Takes all case handlers for a state and assembles them into one coherent,
structured prompt. Non-trivial responsibilities:
- Decide ordering (primary path first, then objections, then errors, then edge cases)
- Smooth language so the prompt reads as one unified document, not a patchwork
- Enforce overall char limit
- Ensure transitions between cases are explicit
- Validate paradigm coherence against MixedDNA principles
"""

import json
import re
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.prompt_loader import load_prompt
from backend.models.schemas import PipelineState, PromptDraft

logger = logging.getLogger(__name__)

# Ordering: primary path categories first, then edge cases
CASE_ORDER = [
    "happy_path",
    "partial_input",
    "wrong_format",
    "ambiguous_input",
    "objection_refusal",
    "multi_intent",
    "repeat_loop",
    "silent_no_response",
    "system_condition",
    "out_of_scope",
    "mid_flow_exit",
    "escalation_trigger",
]


def get_system_prompt() -> str:
    return load_prompt("assembler")


def _build_user_prompt(
    context_schema_json: str,
    state_name: str,
    intent: str,
    handlers: list[dict],
    char_limit: int,
    variables_json: str,
    dna_principles: list[str] | None = None,
) -> str:
    # Sort handlers by canonical order
    def sort_key(h):
        cat = h.get("category", "")
        try:
            return CASE_ORDER.index(cat)
        except ValueError:
            return len(CASE_ORDER)

    sorted_handlers = sorted(handlers, key=sort_key)

    handlers_text = ""
    for h in sorted_handlers:
        handlers_text += f"""
--- CASE: {h['case_name']} ({h['category']}) ---
Condition: {h['condition']}
Tone: {h.get('tone', 'default')}
Transition: {h.get('transition_to', 'none')}
Variables: {', '.join(h.get('variables_used', []))}
Handler:
{h['bot_response']}
"""

    case_names = [h["case_name"] for h in sorted_handlers]

    # Add DNA coherence instructions if available
    dna_section = ""
    if dna_principles:
        dna_lines = "\n".join(f"  - {p}" for p in dna_principles[:20])
        dna_section = f"""

9. PARADIGM DNA COHERENCE: The assembled prompt must comply with these learned principles:
{dna_lines}

   These are non-negotiable architectural principles. The assembled prompt must
   demonstrably embody them in structure, language, and flow.
"""

    return f"""Context Schema:
{context_schema_json}

STATE: {state_name}
INTENT: {intent}
CHARACTER LIMIT: {char_limit} characters (STRICT -- do NOT exceed)

LOCKED VARIABLES:
{variables_json}

CASE HANDLERS TO ASSEMBLE (in recommended order):
{handlers_text}

ASSEMBLY INSTRUCTIONS:
1. ORDERING: Happy path first, then common variations, then edge cases, then escalation last.
2. COHERENCE: The final prompt must read as ONE unified document, not a list of pasted blocks.
   Smooth transitions between sections. Use natural flow language.
3. COMPLETENESS: Every case handler above MUST appear in the final prompt. Do NOT silently
   drop cases to hit the char limit -- if you're over, tighten language, don't remove cases.
4. STRUCTURE: Use clear conditional structure so the bot knows WHEN each case applies.
   "If the user...", "When...", "In cases where..." etc.
5. PERSONA: The entire prompt must consistently reflect the persona from context_schema.
6. VARIABLE FORMAT (MANDATORY): ALL variable references MUST use double curly braces.
   Write {{{{first_name}}}} not [first_name] or bare first_name.
   Scan every handler and convert ANY non-compliant variable references to {{{{var_name}}}} format.
   Use ONLY the locked variables -- never invent new ones.
7. CHARACTER LIMIT: Count characters. If over, compress phrasing. NEVER sacrifice case
   coverage to hit the limit.
8. TRANSITION ROUTING (MANDATORY): You MUST include transition logic using this approach:
   - INLINE ROUTING: Within each case handler, state the transition at the end of the handler.
     Use the format: "→ Transition to [state_name]" or for conditional transitions:
     "If {{{{var}}}} is X → state_a, elif Y → state_b, else → state_c"
   - ROUTING SUMMARY SECTION: At the very END of the assembled prompt, add a clearly labeled
     section titled "## ROUTING" that consolidates ALL transition paths from this state:
     ```
     ## ROUTING
     - Happy path (user confirms) → next_state_name
     - User refuses → handle_objection
     - Escalation triggered → transfer_to_agent
     - Invalid input (retry < 3) → [loop: current_state]
     - Max retries exceeded → fallback_state
     ```
   - Both inline AND the routing summary are required. Inline provides situational context,
     the routing summary provides a quick-reference table for the system.
{dna_section}
Write the complete assembled prompt now. Output ONLY the raw prompt text, no JSON wrapper,
no markdown, no preamble, no explanation."""


async def assemble_prompts(state: PipelineState) -> dict:
    """LangGraph node: assemble case handlers into final prompts for all states."""
    llm = get_llm(max_tokens=3000)

    context_schema = state.get("context_schema", {})
    case_handlers = state.get("case_handlers", [])
    prioritised_cases = state.get("prioritised_cases", [])
    extracted_vars = state.get("extracted_variables", [])
    context_json = json.dumps(context_schema, indent=2)
    variables_json = json.dumps(extracted_vars, indent=2)

    # Extract MixedDNA principles for coherence validation
    mixed_dna = state.get("mixed_dna")
    dna_principles = None
    if mixed_dna and not mixed_dna.get("is_cold_start", True):
        dna_principles = []
        paradigm_names = [
            "structural", "linguistic", "behavioral", "persona",
            "transition", "constraint", "recovery", "rhythm",
        ]
        for p in paradigm_names:
            pp = mixed_dna.get(p, {})
            for pr in pp.get("principles", []):
                dna_principles.append(f"[{p.upper()}] {pr}")
        if not dna_principles:
            dna_principles = None

    if not case_handlers:
        return {
            "drafts": [],
            "progress": "Assembly skipped (no case handlers)",
        }

    # Index prioritised cases by state for intent/char_budget
    pcl_by_state = {}
    for pcl in prioritised_cases:
        pcl_by_state[pcl.get("state_name", "")] = pcl

    drafts = []
    retrieval_contexts = []

    for ch_output in case_handlers:
        state_name = ch_output.get("state_name", "")
        handlers = ch_output.get("handlers", [])
        pcl = pcl_by_state.get(state_name, {})
        intent = pcl.get("intent", "")
        char_limit = pcl.get("total_char_budget", 4500)

        if state_name in ("global_instructions", "global_instruction"):
            char_limit = 2000

        user_prompt = _build_user_prompt(
            context_json, state_name, intent, handlers, char_limit, variables_json,
            dna_principles=dna_principles,
        )

        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=user_prompt),
        ]

        response = await llm.ainvoke(messages)
        prompt_text = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()

        # Track which cases were included
        case_names = [h.get("case_name", "") for h in handlers]

        drafts.append(
            PromptDraft(
                state_name=state_name,
                prompt=prompt_text,
                case_breakdown=case_names,
                retrieved_examples=[],
            ).model_dump()
        )

    logger.info(f"Prompt Assembler: {len(drafts)} prompts assembled")

    return {
        "drafts": drafts,
        "retrieval_contexts": retrieval_contexts,
        "progress": f"Prompts assembled ({len(drafts)} states)",
    }


async def reassemble_prompt(state: PipelineState) -> dict:
    """Regeneration: re-assemble a single state from Case Prioritiser onward.

    Re-runs with user feedback incorporated into the case writer.
    """
    context_schema = state.get("context_schema", {})
    regen_state_name = state.get("regen_state_name", "")
    regen_reason = state.get("regen_reason", "")

    if not regen_state_name:
        return {"error": "No state specified for regeneration"}

    # Find existing handlers for this state
    case_handlers = state.get("case_handlers", [])
    target_handlers = None
    for ch in case_handlers:
        if ch.get("state_name") == regen_state_name:
            target_handlers = ch
            break

    if not target_handlers:
        return {"error": f"No handlers found for state '{regen_state_name}'"}

    # Re-assemble with feedback
    llm = get_llm(max_tokens=3000)
    context_json = json.dumps(context_schema, indent=2)
    variables_json = json.dumps(state.get("extracted_variables", []), indent=2)

    pcl_by_state = {}
    for pcl in state.get("prioritised_cases", []):
        pcl_by_state[pcl.get("state_name", "")] = pcl

    pcl = pcl_by_state.get(regen_state_name, {})
    intent = pcl.get("intent", "")
    char_limit = pcl.get("total_char_budget", 4500)

    # Extract MixedDNA principles — same as main assemble_prompts path
    mixed_dna = state.get("mixed_dna")
    dna_principles = None
    if mixed_dna and not mixed_dna.get("is_cold_start", True):
        dna_principles = []
        paradigm_names = [
            "structural", "linguistic", "behavioral", "persona",
            "transition", "constraint", "recovery", "rhythm",
        ]
        for p in paradigm_names:
            pp = mixed_dna.get(p, {})
            for pr in pp.get("principles", []):
                dna_principles.append(f"[{p.upper()}] {pr}")
        if not dna_principles:
            dna_principles = None

    user_prompt = _build_user_prompt(
        context_json, regen_state_name, intent,
        target_handlers.get("handlers", []), char_limit, variables_json,
        dna_principles=dna_principles,
    )

    if regen_reason:
        user_prompt += f"""

CRITICAL -- REGENERATION FEEDBACK:
The previous assembled prompt was rejected by the user.
Reason: '{regen_reason}'

Address this feedback directly. Do NOT repeat the same approach.
Write a fundamentally different assembly that fixes this issue."""

    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    new_prompt = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()

    case_names = [h.get("case_name", "") for h in target_handlers.get("handlers", [])]

    # Update the specific draft
    drafts = list(state.get("drafts", []))
    for i, d in enumerate(drafts):
        if d["state_name"] == regen_state_name:
            drafts[i] = PromptDraft(
                state_name=regen_state_name,
                prompt=new_prompt,
                case_breakdown=case_names,
                status="pending",
            ).model_dump()
            break

    return {
        "drafts": drafts,
        "regen_state_name": None,
        "regen_reason": None,
        "progress": f"Regenerated prompt for {regen_state_name}",
    }
