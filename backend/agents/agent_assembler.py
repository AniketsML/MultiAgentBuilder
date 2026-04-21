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
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.prompt_loader import load_prompt
from backend.models.schemas import PipelineState, PromptDraft
from backend.kb import sqlite_db

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
    format_examples: list[str] | None = None,
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
Handler:
{h['bot_response']}
"""

    # Add DNA coherence instructions if available
    dna_section = ""
    if dna_principles:
        dna_lines = "\n".join(f"  - {p}" for p in dna_principles[:20])
        dna_section = f"""
PARADIGM DNA COHERENCE: The assembled prompt must comply with these learned principles:
{dna_lines}

   These are non-negotiable architectural principles. The assembled prompt must
   demonstrably embody them in structure, language, and flow.
"""

    # ── Format reference: inject KB/sample prompt examples for structural matching ─
    format_ref_section = ""
    if format_examples:
        refs = []
        for i, ex in enumerate(format_examples[:2], 1):
            refs.append(f"--- FORMAT REFERENCE {i} ---\n{str(ex)[:3000]}")
        format_ref_section = f"""
================================================================================
FORMAT REFERENCE — PRODUCTION PROMPTS (YOUR OUTPUT MUST MATCH THIS STRUCTURE)
================================================================================
These are real production prompts. They might be for different use cases or 
intents, but your assembled prompt MUST replicate their structural format exactly. 
This is the structural DNA you must clone. Key patterns to enforce:

  1. GOAL: line first — one sentence, variables inline, all-caps GOAL label
  2. Capitalized section headers per scenario: "Agreement to Pay", "REFUSAL HANDLING",
     "SPECIAL Scenario Response / Action", "Disclosure:", "Reasoning:"
  3. Numbered sequential probes: "First Probe (Strict):", "Second Probe (Moderate):",
     "Third Probe (Final):" with explicit acceptance criteria per probe
  4. Explicit max attempts: "(Max 3 Probes)", "retry up to 3 times"
  5. Rotation rules inline: "Rotate categories... Never repeat."
  6. Validation math: "Accept date within {{{{due_date}}}} + 2 days", "must fall within current month"
  7. Arrow branches: "If Yes → @state_name; If No → @other_state"
  8. Direct imperative language: "Inform {{{{user_name}}}} that...", "Ask {{{{user_name}}}} to provide..."
  9. SPECIAL Scenario as a named section aggregating all special cases

  ABSOLUTELY FORBIDDEN in output:
  - Soft paragraph prose: "greet warmly", "reassure gently", "acknowledge respectfully"
  - Generic empathy filler before operational instructions
  - Narrative case descriptions instead of direct instructions
  - 'Case 1:', 'Handler:', 'Block A:' labels

{''.join(refs)}
================================================================================
"""

    # ── Global instructions: inject dedicated header ───────────────────────────
    is_gi = state_name in ("global_instructions", "global_instruction")
    gi_header = f"""ASSEMBLING: global_instructions — PERMANENT BEHAVIORAL CONTRACT
This is NOT a conversational state. It governs EVERY other state system-wide.
This is the most important prompt in the output — write it as a BINDING OPERATIONAL MANDATE.

MANDATORY RULES for global_instructions:
- ZERO @state_name transitions — add none
- NO routing, no state-specific scenarios, no conditional case handling
- MUST cover ALL of:
    1. Identity + persona (name, traits, voice, anti-examples of wrong tone)
    2. Tone mandate (primary register + exact conditions for shifts + prohibited registers)
    3. Behavioral laws (every guardrail as NEVER... absolute prohibitions)
    4. Universal escalation protocol (exact trigger phrases + scripted immediate pivot)
    5. Universal fallback protocol (Attempt 1 / Attempt 2 / Attempt 3 with tone at each)
    6. Call integrity rules (identity verification, silence, mid-call exit)
- Write what the bot WILL DO — not what it "should" consider
- Character limit: {char_limit} — compress ruthlessly, lose no rule
- Start with: Goal: [one precise sentence — the system-wide behavioral mandate]

""" if is_gi else ""

    return f"""{gi_header}{format_ref_section}Context Schema:
{context_schema_json}

STATE: {state_name}
INTENT: {intent}
CHARACTER LIMIT: {char_limit} characters (STRICT — do NOT exceed)

LOCKED VARIABLES (real backend-passed fields — use ONLY these, never invent new ones):
{variables_json}

CASE HANDLERS TO ASSEMBLE:
{handlers_text}

ASSEMBLY INSTRUCTIONS:
1. FORMAT PRIMACY (HIGHEST PRIORITY): The assembled prompt MUST match the structural format
   of the FORMAT REFERENCE above. Use the same section-header style, probe numbering,
   rotation rules, and validation math shown in those examples.
   If no format reference is provided, use the dense operational style: numbered sections,
   capitalized headers, explicit probe counts, arrow branching — NOT soft prose.
2. GOAL LINE (MANDATORY FIRST LINE): GOAL: [one precise sentence — what this state achieves]
3. STRUCTURAL FORMAT: Use capitalized section names per scenario:
   'Agreement to Pay:', 'REFUSAL HANDLING (Max N Probes):', 'SPECIAL Scenario Response / Action:'
   Numbered sub-probes: 'First Probe (Strict):', 'Second Probe (Moderate):', 'Third Probe (Final):'
   Arrow branches: 'If Yes → @state_name; If No → @other_state'
4. LANGUAGE: Direct imperative ONLY. 'Inform {{{{user_name}}}} that...', 'Ask {{{{user_name}}}} to...'
   NEVER: 'greet warmly', 'reassure', 'acknowledge respectfully', soft empathy filler.
5. PRESERVE DECISION-TREE DEPTH: All nested sub-conditions, numbered probes, validation rules,
   rotation rules, explicit attempt counts MUST survive assembly. Do NOT flatten.
6. PERSONA: One consistent voice throughout. No drift between sections.
7. VARIABLE FORMAT (ABSOLUTE): {{{{variable_name}}}} — double braces, snake_case only.
   ONLY locked variables. Correct: {{{{user_name}}}}, {{{{lender_name}}}}, {{{{due_date}}}}, {{{{emi_amount}}}}
8. CHARACTER LIMIT: Compress phrasing if over. NEVER drop cases or @transitions.
9. INLINE TRANSITIONS — ABSOLUTE RULE:
   - Always: → @state_name inline at end of the branch it belongs to
   - Conditional: 'If confirmed → @payment_collection; if declined → @call_closed'
   - NEVER 'transition to state_name' — @state_name format ONLY
   - NEVER a Routing/Transitions/Navigation section — ABSOLUTELY FORBIDDEN
   - global_instructions: ZERO @state_name transitions
10. COMPLETENESS: Every handler MUST appear. Drop nothing.
{dna_section}
Write the complete assembled prompt now. Output ONLY the raw prompt text — no JSON, no markdown, no preamble."""



async def assemble_prompts(state: PipelineState) -> dict:
    """LangGraph node: assemble case handlers into final prompts for all states."""
    llm = get_llm(max_tokens=4096)

    context_schema = state.get("context_schema", {})
    case_handlers = state.get("case_handlers", [])
    prioritised_cases = state.get("prioritised_cases", [])
    case_learning_contexts = state.get("case_learning_contexts", [])
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

    # Extract format_examples from context_schema.raw_examples
    # These are the user's uploaded sample prompts — the ground truth for output format
    format_examples = None
    raw_ex = context_schema.get("raw_examples")
    if isinstance(raw_ex, list) and raw_ex:
        format_examples = [str(e) for e in raw_ex if e]
    elif isinstance(raw_ex, str) and raw_ex.strip():
        format_examples = [raw_ex]

    if not case_handlers:
        return {
            "drafts": [],
            "progress": "Assembly skipped (no case handlers)",
        }

    # Index prioritised cases by state for intent/char_budget
    pcl_by_state = {}
    for pcl in prioritised_cases:
        pcl_by_state[pcl.get("state_name", "")] = pcl

    # Index case learning by state for fetching usecase-specific DB prompt examples
    cl_by_state = {}
    for cl in case_learning_contexts:
        cl_by_state[cl.get("state_name", "")] = cl

    drafts = []
    retrieval_contexts = []

    completed_count = 0
    total = len(case_handlers)

    async def _process(ch_output):
        nonlocal completed_count
        state_name = ch_output.get("state_name", "")
        await sqlite_db.update_run_progress(
            state["run_id"],
            f"Prompt Assembly: processing {state_name}..."
        )
        handlers = ch_output.get("handlers", [])
        pcl = pcl_by_state.get(state_name, {})
        intent = pcl.get("intent", "")
        char_limit = pcl.get("total_char_budget", 4500)

        # Prioritize use-case specific prompt from DB over global fallback
        cl = cl_by_state.get(state_name, {})
        state_format_examples = format_examples
        if cl and cl.get("raw_prompts"):
            state_format_examples = cl["raw_prompts"]

        user_prompt = _build_user_prompt(
            context_json, state_name, intent, handlers, char_limit, variables_json,
            dna_principles=dna_principles,
            format_examples=state_format_examples,
        )

        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=user_prompt),
        ]

        response = await llm.ainvoke(messages)
        prompt_text = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()

        # Track which cases were included
        case_names = [h.get("case_name", "") for h in handlers]

        draft = PromptDraft(
            state_name=state_name,
            prompt=prompt_text,
            case_breakdown=case_names,
            retrieved_examples=[],
        )

        completed_count += 1
        await sqlite_db.update_run_progress(
            state["run_id"],
            f"Prompt Assembly {completed_count}/{total}: finished {state_name}"
        )
        return draft


    tasks = [_process(ch) for ch in case_handlers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            logger.error(f"[Assembler] Error in async assembly: {res}")
            continue
        drafts.append(res.model_dump())

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

    # Extract format_examples from context_schema.raw_examples
    format_examples = None
    raw_ex = context_schema.get("raw_examples")
    if isinstance(raw_ex, list) and raw_ex:
        format_examples = [str(e) for e in raw_ex if e]
    elif isinstance(raw_ex, str) and raw_ex.strip():
        format_examples = [raw_ex]

    # Prioritize use-case specific prompt from DB over global fallback
    case_learning_contexts = state.get("case_learning_contexts", [])
    for cl in case_learning_contexts:
        if cl.get("state_name") == regen_state_name and cl.get("raw_prompts"):
            format_examples = cl["raw_prompts"]
            break

    user_prompt = _build_user_prompt(
        context_json, regen_state_name, intent,
        target_handlers.get("handlers", []), char_limit, variables_json,
        dna_principles=dna_principles,
        format_examples=format_examples,
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
