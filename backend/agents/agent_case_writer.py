"""Case Writer — new agent.

For each case in the PrioritisedCaseList, writes a self-contained handling block:
- What the bot says
- What variables it uses
- What condition triggers this case
- What state it transitions to afterward

Draws on CaseLearningContext to learn handling strategy (not words).
Now also receives MixedDNA as hard constraints — paradigm principles
that each handler must respect.

Enforces: only locked variables, char budget awareness, persona consistency,
and paradigm compliance.
"""

import json
import logging
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.json_parser import extract_json, json_ainvoke_with_retry
from backend.utils.prompt_loader import load_prompt
from backend.models.schemas import (
    PipelineState, CaseHandler, CaseWriterOutput,
)
from backend.kb import sqlite_db

logger = logging.getLogger(__name__)


def get_system_prompt() -> str:
    return load_prompt("case_writer")


GLOBAL_INSTRUCTIONS_NAMES = {"global_instructions", "global_instruction"}


def _build_global_instructions_writer_prompt(
    context_schema_json: str,
    prioritised_case_list: dict,
) -> str:
    """Dedicated prompt for writing global_instructions — outputs JSON handler format."""
    cases = prioritised_case_list.get("cases", [])
    extracted = "\n".join(
        f"- {c.get('case_name', '')}: {c.get('handling_hint', '')}"
        for c in cases
    )
    return f"""Write the GLOBAL INSTRUCTIONS behavioral contract for a conversational voice bot.

This is the PERMANENT system-level mandate that governs EVERY state of the bot.
It defines WHO the bot is, HOW it must always behave, WHAT it must never do,
and how it universally recovers from breakdown — for the ENTIRE call duration.

Context Schema:
{context_schema_json}

System-wide rules extracted from state decomposition:
{extracted}

The global_instructions MUST contain ALL of the following — no omissions, no vagueness:

1. IDENTITY & PERSONA (specific, not generic)
   - Bot name, role, exact personality traits, communication style
   - Sentence rhythm, directness level, formality, use of first name
   - What the bot NEVER sounds like (give anti-examples)

2. TONE MANDATE
   - Primary emotional register + precise conditions that shift it
   - Example: "Professional and composed at all times; shift to empathetic warmth when
     user expresses distress or hardship; NEVER match aggression with aggression"
   - Explicitly prohibited registers (threatening, groveling, passive)

3. BEHAVIORAL LAWS (apply in EVERY state, zero exceptions)
   - Every guardrail as an absolute prohibition (NEVER...)
   - Personalization: address user as {{{{user_name}}}}, cite {{{{lender_name}}}},
     reference {{{{due_date}}}} and {{{{emi_amount}}}} when relevant
   - Prohibited phrases and compliance rules

4. UNIVERSAL ESCALATION PROTOCOL
   - Exact trigger phrases/signals (legal threat, abusive language, medical emergency,
     explicit self-harm mention, third-party interference)
   - Bot's IMMEDIATE behavior upon trigger (specific scripted pivot, NOT 'hand off')
   - Escalation rules here SUPERSEDE all state-level handling

5. UNIVERSAL FALLBACK PROTOCOL (3-strike)
   - Attempt 1: How bot rephrases and re-engages on first non-parse
   - Attempt 2: How it simplifies and changes approach on second failure
   - Attempt 3: Final action — escalate or close gracefully
   - Tone at each level: patient → patient but firm → decisive

6. CALL INTEGRITY RULES
   - Identity verification: when required, how enforced, what to do when refused
   - Silence protocol: dead-air threshold, re-engagement language, escalation trigger
   - Mid-call exit: exact closure behavior if user hangs up or demands to stop

OUTPUT FORMAT — MANDATORY JSON:
Return ONLY a valid JSON object in this exact format:
{{
  "state_name": "global_instructions",
  "handlers": [
    {{
      "case_name": "global_instructions_contract",
      "category": "happy_path",
      "condition": "Applied system-wide across every state of the conversation",
      "bot_response": "[Write the complete global instructions here as a single dense operational text block. Start with: Goal: [one precise sentence]. Then cover all 6 sections above with specific, decisive language. Max 2000 characters. NO @state_name transitions. NO routing. Write what the bot WILL DO, not guidelines.]",
      "variables_used": [],
      "tone": "Decisive, operational, binding",
      "char_count": 0
    }}
  ],
  "total_char_count": 0
}}"""


def _build_user_prompt(
    context_schema_json: str,
    prioritised_case_list: dict,
    case_learning: dict | None,
    variables_json: str,
    mixed_dna_principles: list[str] | None = None,
    raw_kb_examples: list[str] | None = None,
) -> str:
    state_name = prioritised_case_list.get("state_name", "")
    intent = prioritised_case_list.get("intent", "")
    char_budget = prioritised_case_list.get("total_char_budget", 4500)

    # global_instructions uses a completely separate, rules-only writer
    if state_name in GLOBAL_INSTRUCTIONS_NAMES:
        return _build_global_instructions_writer_prompt(
            context_schema_json, prioritised_case_list
        )

    # Only include cases with action="keep"
    kept_cases = [c for c in prioritised_case_list.get("cases", []) if c.get("action") == "keep"]

    cases_json = json.dumps(kept_cases, indent=2)

    # Format KB case learning — preserve full depth of sub-conditions
    kb_section = ""
    if case_learning and case_learning.get("learned_cases"):
        strategies = []
        for lc in case_learning["learned_cases"]:
            strategy_text = lc['handling_strategy'][:600]
            variables = ", ".join(lc.get('variables_used', [])[:5])
            anti = lc.get('anti_patterns', '')
            anti_text = f" | AVOID: {str(anti)[:200]}" if anti else ""
            strategies.append(
                f"  [{lc['case_category']}]: {strategy_text}"
                f" | tone: {lc.get('tone_approach', 'N/A')}"
                f" | vars: {variables}"
                f"{anti_text}"
            )
        kb_section = f"""
LEARNED HANDLING STRATEGIES (from KB — learn the DEPTH and BRANCHING approach, not the words):
These strategies show HOW real prompts handle this domain. Notice their nested sub-cases, validation
steps, specific conditions, and branching logic. Your handlers must match this depth.
{chr(10).join(strategies[:30])}
"""

    # Raw KB reference examples — show the actual production prompt structure
    kb_reference_section = ""
    if raw_kb_examples:
        refs = []
        for i, ex in enumerate(raw_kb_examples[:2], 1):
            refs.append(f"--- KB REFERENCE PROMPT {i} ---\n{ex[:3000]}")
        kb_reference_section = f"""
================================================================================
FORMAT REFERENCE — PRODUCTION KB PROMPTS (MANDATORY TO MATCH THIS STRUCTURE)
================================================================================
These are real production prompts from the knowledge base. They might be for 
different use cases or intents, but you MUST replicate their structural format 
exactly. This is the structural DNA you must clone:

  STRUCTURAL PATTERNS TO COPY:
  - GOAL: line as the first line with variables inline
  - Capitalized section headers for each major scenario (e.g. "Agreement to Pay",
    "REFUSAL HANDLING (Max 3 Probes)", "SPECIAL Scenario Response / Action")
  - Numbered sequential probes: "First Probe (Strict):", "Second Probe (Moderate):",
    "Third Probe (Final):" with explicit acceptance rules per probe
  - Explicit max-attempt counts: "(Max 3 Probes)", "retry up to 3 times"
  - Rotation rules: "Rotate categories... Never repeat."
  - Validation math: "Accept date within {{{{due_date}}}} + 2 days"
  - Arrow-branching: "If Yes → @state; If No → @other_state"
  - SPECIAL Scenario section with named scenarios as sub-entries
  - Direct imperative instructions: "Inform {{{{user_name}}}} that...", "Ask {{{{user_name}}}} to provide..."
  - State/function calls inline: "→ @state_name"
  NO soft paragraph prose. NO empathy-first language. ONLY operational instructions.

{''.join(refs)}
================================================================================
"""
    dna_section = ""
    if mixed_dna_principles:
        dna_lines = "\n".join(f"  - {p}" for p in mixed_dna_principles)
        dna_section = f"""
PARADIGM DNA CONSTRAINTS (these are learned architectural principles -- you MUST follow them):
{dna_lines}

Every handler you write must demonstrably comply with these principles.
"""

    return f"""Context Schema:
{context_schema_json}

STATE: {state_name}
INTENT: {intent}
TOTAL CHARACTER BUDGET: {char_budget} characters (for ALL handlers combined, leave ~500 for assembly overhead)

LOCKED VARIABLES — these are real values passed from the backend at runtime.
Use ONLY these. Never invent, hallucinate, or generalize. Always wrap in double curly braces.
Think of these as: user_name, lender_name, due_date, emi_amount — actual data fields.
{variables_json}
{kb_reference_section}{kb_section}{dna_section}

CASES TO WRITE HANDLERS FOR:
{cases_json}

For EACH case, write a self-contained handling block as a JSON object:
{{
  "case_name": "...",
  "category": "...",
  "condition": "The exact user input, behavioral signal, or system condition that triggers this handler — be precise, not generic",
  "bot_response": "MANDATORY FORMAT — COPY THE KB REFERENCE STRUCTURE:\n\nYour bot_response MUST use the same structural format as the KB reference prompts above.\nSpecifically:\n1. If this is a primary scenario: start with a capitalized section name, e.g. 'Agreement to Pay:' or 'REFUSAL HANDLING (Max 3 Probes)'\n2. Use numbered sequential probes where applicable: 'First Probe (Strict):', 'Second Probe (Moderate):', 'Third Probe (Final):'\n3. State explicit max attempt counts: '(Max 3 Probes)', 'retry up to 3 times'\n4. State rotation rules where applicable: 'Rotate through... Never repeat'\n5. State validation rules with math: 'Accept date within {{{{due_date}}}} + 2 days'\n6. Use arrow-branch notation: 'If Yes → @state_name; If No → @other_state'\n7. Direct imperative language ONLY: 'Inform {{{{user_name}}}} that...', 'Ask {{{{user_name}}}} to provide...'\n8. Group SPECIAL scenarios as a named section at the end when applicable\nDO NOT: write soft paragraph prose, write 'greet warmly', 'reassure', 'acknowledge respectfully'\nDO NOT: start with empathy — start with the operational instruction\nVariables: {{{{variable_name}}}} always. @state_name for all inline transitions.",
  "variables_used": ["only from the locked list above — real backend fields"],
  "tone": "tone used in this handler",
  "char_count": 0
}}

RULES:
- SELF-CONTAINED: Each handler must be fully readable and actionable without referencing other handlers
- PERSONA MATCH: Every handler reflects the persona from context_schema.persona — no persona drift
- DECISION-TREE DEPTH IS MANDATORY:
  BAD: "If the user is busy, acknowledge and offer a callback."
  GOOD: "If the user states they are busy (e.g. driving, at work, in a meeting):\n    - Acknowledge immediately without restating the call purpose.\n    - Ask: 'What time today or tomorrow works best to call you back?'\n    - Sub-case A: User gives specific time → confirm the time using {{{{user_name}}}}, log callback, @callback_scheduled\n    - Sub-case B: User says 'don't call again' or refuses time → acknowledge stance, state next steps per policy, @call_closed\n    - Sub-case C: User gives vague time ('later', 'sometime') → probe once more: 'Just to make sure I reach you, would morning or afternoon work better?' → if still vague → @callback_scheduled with note UNCONFIRMED_TIME"
- TONE BY CASE TYPE:
  - objection_refusal: empathetic, validate concern, pivot without pressure
  - wrong_format / repeat_loop: patient, provide explicit example, never condescending
  - escalation_trigger: calm, de-escalate, never match aggression
  - system_condition: clear, solution-focused, no filler
  - happy_path: match persona primary tone
  - silent_no_response: gentle re-engagement, not accusatory
  - mid_flow_exit: acknowledge intent, confirm closure, clean exit
- VARIABLE FORMAT (ABSOLUTE): {{{{variable_name}}}} — double braces always, snake_case.
  NEVER use [variable_name], (variable_name), or bare variable_name.
  ONLY variables from the locked list — if a variable isn't listed, do NOT use it.
- INLINE TRANSITION FORMAT (MANDATORY): Every branch must end with @next_state_name
  Written inline at the end of the branch, as part of the prose: '@payment_collection'
  For conditional routing: 'if confirmed → @payment_collection; if declined → @call_closed'
  NEVER create a separate transition_to field. NEVER write 'transition to state_name' — use @state_name ONLY.
- EDGE CASES MUST BE EXPLICIT: For every case, identify and handle the 2-3 most likely edge sub-conditions,
  not just the clean path. Example: if the case is 'user gives payment date', also handle:
  - date is in the past
  - date is more than 30 days away
  - user is vague about the date
- NORMAL CASES must be crystal clear and complete — the happy path should be the most detailed handler
- Keep each handler tight but COMPLETE — depth over brevity. Compress only if over budget.

Return ONLY a JSON object:
{{
  "state_name": "{state_name}",
  "handlers": [...],
  "total_char_count": 0
}}"""


def _coerce_handler(raw: dict) -> dict:
    """Normalize LLM output for CaseHandler."""
    raw.setdefault("case_name", "")
    raw.setdefault("category", "happy_path")
    raw.setdefault("condition", "")
    raw.setdefault("bot_response", "")
    raw.setdefault("transition_to", "")
    raw.setdefault("tone", "")

    if raw.get("variables_used") is None:
        raw["variables_used"] = []
    elif isinstance(raw["variables_used"], str):
        raw["variables_used"] = [raw["variables_used"]] if raw["variables_used"].strip() else []

    # Calculate char count
    raw["char_count"] = len(raw.get("bot_response", ""))

    return raw


async def write_case_handlers(state: PipelineState) -> dict:
    """LangGraph node: write case handlers for all states."""
    llm = get_llm(max_tokens=4096)

    context_schema = state.get("context_schema", {})
    prioritised_cases = state.get("prioritised_cases", [])
    case_learning_contexts = state.get("case_learning_contexts", [])
    extracted_vars = state.get("extracted_variables", [])
    context_json = json.dumps(context_schema, indent=2)
    variables_json = json.dumps(extracted_vars, indent=2)

    # Extract MixedDNA principles for injection
    mixed_dna = state.get("mixed_dna")
    dna_principles = None
    if mixed_dna and not mixed_dna.get("is_cold_start", True):
        # Flatten all paradigm principles into a tagged list
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

    if not prioritised_cases:
        return {
            "case_handlers": [],
            "progress": "Case writing skipped (no prioritised cases)",
        }

    # Index case learning by state name
    cl_by_state = {}
    for clc in case_learning_contexts:
        cl_by_state[clc.get("state_name", "")] = clc

    all_handlers = []

    completed_count = 0
    total = len(prioritised_cases)

    async def _process(pcl):
        nonlocal completed_count
        state_name = pcl.get("state_name", "")
        intent = pcl.get("intent", "")
        char_limit = pcl.get("total_char_budget", 4500)
        
        await sqlite_db.update_run_progress(
            state["run_id"],
            f"Case Handlers: processing {state_name}..."
        )
        case_learning = cl_by_state.get(state_name)

        # Extract raw prompt texts from KB learning for direct structural reference.
        # Priority: raw_prompts from case_learning -> raw_examples from context_schema
        # (context_schema.raw_examples are the user's uploaded sample prompts verbatim)
        raw_kb_examples = None
        if case_learning and case_learning.get("raw_prompts"):
            raw_kb_examples = case_learning["raw_prompts"]
        elif context_schema.get("raw_examples"):
            # Fall back to sample prompts uploaded by the user — these define the target style
            raw_examples = context_schema["raw_examples"]
            if isinstance(raw_examples, list):
                raw_kb_examples = raw_examples
            elif isinstance(raw_examples, str) and raw_examples.strip():
                raw_kb_examples = [raw_examples]

        user_prompt = _build_user_prompt(
            context_schema_json=context_json,
            prioritised_case_list=pcl,
            case_learning=case_learning,
            variables_json=variables_json,
            mixed_dna_principles=dna_principles,
            raw_kb_examples=raw_kb_examples,
        )

        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=user_prompt),
        ]

        parsed = await json_ainvoke_with_retry(llm, messages)
        if isinstance(parsed, list):
            parsed = {"handlers": parsed}

        # Coerce handlers
        handlers = []
        for h in parsed.get("handlers", []):
            handlers.append(_coerce_handler(h))

        total_chars = sum(h["char_count"] for h in handlers)

        output = CaseWriterOutput(
            state_name=state_name,
            handlers=handlers,
            total_char_count=total_chars,
        )
        
        completed_count += 1
        await sqlite_db.update_run_progress(
            state["run_id"],
            f"Case Handlers {completed_count}/{total}: finished {state_name}"
        )
        return output

    tasks = [_process(pcl) for pcl in prioritised_cases]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            logger.error(f"[CaseWriter] Error in async case writer: {res}")
            continue
        all_handlers.append(res.model_dump())

    total_handlers = sum(len(h["handlers"]) for h in all_handlers)

    logger.info(
        f"Case Writer: {len(all_handlers)} states, "
        f"{total_handlers} total handlers written"
    )

    return {
        "case_handlers": all_handlers,
        "progress": f"Case handlers written ({total_handlers} handlers across {len(all_handlers)} states)",
    }
