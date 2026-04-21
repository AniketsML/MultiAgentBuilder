"""Agent 2 — State Decomposer (heavily upgraded from State Planner).

Takes one state at a time + CaseLearningContext from KB Case Learner.
Breaks the state into an exhaustive case map using the canonical taxonomy.
Variables are extracted here as a natural by-product of case decomposition.
"""

import json
import logging
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage

from backend.utils.json_parser import extract_json, json_ainvoke_with_retry

# ... (I need to be careful with imports)
from backend.agents.claude_client import get_llm
from backend.models.schemas import (
    PipelineState, StateDecomposition, CaseSpec, VariableSchema,
)
from backend.utils.prompt_loader import load_prompt
from backend.kb import sqlite_db

logger = logging.getLogger(__name__)

# Canonical case taxonomy
CASE_TAXONOMY = [
    "happy_path",
    "partial_input",
    "objection_refusal",
    "ambiguous_input",
    "wrong_format",
    "system_condition",
    "repeat_loop",
    "out_of_scope",
    "silent_no_response",
    "escalation_trigger",
    "multi_intent",
    "mid_flow_exit",
]


def get_system_prompt() -> str:
    return load_prompt("agent2")


GLOBAL_INSTRUCTIONS_NAMES = {"global_instructions", "global_instruction"}


def _build_global_instructions_prompt(
    context_schema_json: str,
    raw_text: str,
) -> str:
    """Special prompt for global_instructions — the PERMANENT behavioral system mandate."""
    return f"""You are decomposing the GLOBAL INSTRUCTIONS state for a conversational voice bot.

GLOBAL INSTRUCTIONS is NOT a conversational state. It is the PERMANENT behavioral contract
that governs the bot across EVERY state in the conversation. It defines identity, tone laws,
absolute prohibitions, universal escalation protocols, and recovery behavior.

Original Context Document:
---
{raw_text[:3000]}
---

Extracted Context Schema:
{context_schema_json}

Extract ALL of the following with maximum specificity — do NOT be vague:

1. IDENTITY & PERSONA
   - Bot name, role title, and exact personality traits with domain-specific detail
   - Communication style: sentence length, use of first names, formality level, rhythm
   - What the bot NEVER sounds like (anti-example: threats, groveling, over-apologizing)
   - How the bot refers to itself and the user at all times

2. TONE MANDATE
   - Primary tone register + exact conditions that trigger variation
   - Example: "Professional and composed always; shift to empathetic warmth when user expresses
     financial hardship or distress; NEVER match aggression; NEVER use threatening language"
   - Explicitly prohibited emotional registers

3. BEHAVIORAL LAWS (absolute rules — apply in every state without exception)
   - Every guardrail from the context schema as a binding prohibition
   - Personalization rules: when to use {{{{user_name}}}}, {{{{lender_name}}}}, {{{{due_date}}}}
   - Language/format constraints: bilingual rules, prohibited phrases, confirmation requirements
   - What constitutes a legal/compliance violation the bot must avoid

4. UNIVERSAL ESCALATION PROTOCOL
   - Exact trigger phrases/signals that force escalation from ANY state
     (e.g., legal threats, abuse, medical emergency, explicit distress signals)
   - Exact immediate behavior upon trigger: what the bot says BEFORE ending, NOT just "hand off"
   - global_instructions escalation rules SUPERSEDE all state-level handling

5. UNIVERSAL FALLBACK PROTOCOL (3-strike structure)
   - Attempt 1: How the bot rephrases and re-engages on first non-parse
   - Attempt 2: How it simplifies and tries differently on second failure
   - Attempt 3: Final action before forced exit or escalation
   - Recovery language register at each level (patient, then firm, then decisive)

6. CALL INTEGRITY RULES
   - Identity verification: when required, how enforced, what happens on refusal
   - Silence protocol: dead-air threshold, re-engagement language, escalation trigger
   - Mid-call exit: if user disconnects or demands to stop, exact closure behavior

STRICT EXTRACTION RULES:
- Extract ONLY system-wide rules that apply equally in EVERY state
- DO NOT extract any state-specific handling, routing, or case scenarios
- DO NOT include @state_name transitions — global_instructions has ZERO transitions
- Be operationally specific: write what the bot WILL DO, not what it should consider

Return ONLY a JSON object:
{{
  "state_name": "global_instructions",
  "intent": "Permanent behavioral contract: defines who the bot is, how it always behaves, what it must never do, and how it universally recovers — across all states",
  "cases": [
    {{
      "case_name": "identity_and_persona",
      "category": "happy_path",
      "description": "Core identity, personality traits, and communication style of the bot",
      "handling_hint": "Extracted persona: name, voice, formality, first-name usage, anti-examples...",
      "required_variables": [],
      "transition_to": "",
      "tone_guidance": "Defines the universal tone mandate"
    }},
    {{
      "case_name": "behavioral_laws_and_guardrails",
      "category": "system_condition",
      "description": "Absolute prohibitions and compliance rules binding in every state",
      "handling_hint": "Every guardrail extracted from context schema as explicit DO NOT rules",
      "required_variables": [],
      "transition_to": "",
      "tone_guidance": "Firm, non-negotiable"
    }},
    {{
      "case_name": "universal_escalation_protocol",
      "category": "escalation_trigger",
      "description": "Triggers and immediate behavior for universal escalation from any state",
      "handling_hint": "Trigger keywords + exact bot pivot behavior before handoff",
      "required_variables": [],
      "transition_to": "",
      "tone_guidance": "Calm, non-confrontational, de-escalating"
    }},
    {{
      "case_name": "universal_fallback_protocol",
      "category": "repeat_loop",
      "description": "3-attempt recovery sequence for any unrecognized input across all states",
      "handling_hint": "Attempt 1 rephrase → Attempt 2 simplify → Attempt 3 final action",
      "required_variables": [],
      "transition_to": "",
      "tone_guidance": "Patient → patient but firm → decisive"
    }}
  ],
  "extracted_variables": [],
  "dependencies": [],
  "tags": ["persona", "guardrails", "global", "permanent", "behavioral_contract"]
}}"""


def _build_user_prompt(
    context_schema_json: str,
    state_name: str,
    state_names: list[str],
    raw_text: str,
    case_learning: dict | None = None,
) -> str:
    """Build the decomposition prompt for one state."""

    # global_instructions uses a completely separate, tightly scoped prompt
    if state_name in GLOBAL_INSTRUCTIONS_NAMES:
        return _build_global_instructions_prompt(context_schema_json, raw_text)

    names_str = "\n".join(f"- {name}" for name in state_names)

    # Format KB case knowledge if available
    kb_section = ""
    if case_learning and case_learning.get("learned_cases"):
        kb_cases = []
        for lc in case_learning["learned_cases"]:
            sub_cond_text = ""
            sub_conds = lc.get('sub_conditions', [])
            if sub_conds:
                sub_cond_text = f" | sub-conditions: {', '.join(sub_conds[:5])}"
            kb_cases.append(
                f"  - [{lc['case_category']}]: {lc['handling_strategy'][:600]}"
                f"{sub_cond_text}"
                f" (vars: {', '.join(lc.get('variables_used', []))})"
            )
        kb_section = f"""
KB CASE KNOWLEDGE (from similar prompts — learn the DEPTH and BRANCHING STRUCTURE):
These are concrete handling patterns from real production prompts. Use the sub-conditions
and branching logic to generate cases with equivalent depth for this state.
{chr(10).join(kb_cases)}

Anti-patterns to avoid:
{chr(10).join(f'  - {ap}' for ap in case_learning.get('anti_patterns', []))}

IMPORTANT: Notice how each category above has nested sub-cases. Your case decomposition
must have the same level of branching specificity — not flat descriptions.
"""

    taxonomy_str = "\n".join(f"  - {t}" for t in CASE_TAXONOMY)

    return f"""Original Context Document:
---
{raw_text[:3000]}
---

Extracted Context Schema:
{context_schema_json}

All state names in this flow:
{names_str}

STATE TO DECOMPOSE: {state_name}
{kb_section}

CANONICAL CASE TAXONOMY (consider ALL of these for EVERY state):
{taxonomy_str}

Decompose this state into an exhaustive case map covering the happy path AND all realistic
edge cases. For EACH case, provide:
- case_name: descriptive snake_case name
- category: one of the 12 canonical categories above
- description: PRECISELY what user input, signal, or system condition triggers this case.
  NOT generic. Example: "User states they cannot pay because they lost their job" not "user refuses".
- handling_hint: The exact behavioral approach — what the bot should do step by step,
  including sub-conditions. Format as decision-tree steps, not a vague description.
  Include: (1) first bot action, (2) if/else branches, (3) what happens at each branch.
- required_variables: ONLY real backend-passed data fields the bot needs for this case.
  Think: user_name, lender_name, due_date, emi_amount, account_number, loan_id, etc.
  Do NOT invent abstract variables. If the case doesn’t need backend data, use [].
- transition_to: exact state name from the list above. For conditional transitions,
  describe ALL branches: "if X → @state_a; if Y → @state_b"
- tone_guidance: specific tone for this case type (not generic 'professional'—be exact)

VARIABLE RULES (MANDATORY):
- ALL variable references in handling_hint text MUST use: {{{{variable_name}}}}
- Examples: {{{{user_name}}}}, {{{{due_date}}}}, {{{{emi_amount}}}}, {{{{lender_name}}}}
- In extracted_variables list: store name WITHOUT braces ("user_name", "due_date")
- ONLY extract variables that the bot genuinely needs at runtime from the backend
- NEVER invent variables not grounded in the domain context

TRANSITION FORMAT RULES (MANDATORY):
- All transitions use @state_name format inline within handling_hint
- Example: "If user confirms → @payment_collection; if user refuses → @objection_handling"
- transition_to field: use plain state_name (no @), or describe conditional routing as:
  "confirmed: payment_collection | refused: objection_handling"

EDGE CASE REQUIREMENT:
- For EVERY happy_path case, also define at minimum:
  (a) what happens if required data is missing or invalid
  (b) what happens if user acknowledges but does not act
  (c) what happens if user gives an ambiguous or partial response
- Each of these becomes a separate case entry with the correct category.

Respond with ONLY a JSON object:
{{
  "state_name": "{state_name}",
  "intent": "one precise sentence describing the single goal of this state",
  "cases": [...],
  "extracted_variables": [{{"name": "...", "description": "...", "type": "..."}}],
  "dependencies": ["state names that must occur before this"],
  "tags": ["2-4 tags"]
}}"""


def _coerce_decomposition(raw: dict) -> dict:
    """Normalize LLM output for StateDecomposition."""
    # Ensure cases is a list
    cases = raw.get("cases", [])
    if isinstance(cases, dict):
        cases = list(cases.values())

    coerced_cases = []
    for c in cases:
        if isinstance(c, dict):
            # Ensure category is canonical
            cat = c.get("category", "happy_path")
            if cat not in CASE_TAXONOMY:
                cat = "happy_path"
            c["category"] = cat

            # Ensure list fields
            for field in ("required_variables",):
                val = c.get(field)
                if val is None:
                    c[field] = []
                elif isinstance(val, str):
                    c[field] = [val] if val.strip() else []

            # Ensure string fields
            for field in ("case_name", "description", "handling_hint", "transition_to", "tone_guidance"):
                if c.get(field) is None:
                    c[field] = ""

            coerced_cases.append(c)

    raw["cases"] = coerced_cases

    # Ensure variables
    variables = raw.get("extracted_variables", [])
    if isinstance(variables, dict):
        variables = list(variables.values())
    coerced_vars = []
    for v in variables:
        if isinstance(v, dict) and "name" in v:
            v.setdefault("description", "")
            v.setdefault("type", "string")
            coerced_vars.append(v)
    raw["extracted_variables"] = coerced_vars

    # Ensure list fields
    for field in ("dependencies", "tags"):
        val = raw.get(field)
        if val is None:
            raw[field] = []
        elif isinstance(val, str):
            raw[field] = [val] if val.strip() else []

    raw.setdefault("intent", "")
    raw.setdefault("state_name", "")

    return raw


async def decompose_states(state: PipelineState) -> dict:
    """LangGraph node: decompose all states into exhaustive case maps."""
    llm = get_llm(max_tokens=4096)

    context_json = json.dumps(state["context_schema"], indent=2)
    state_names = state["state_names"]
    raw_text = state.get("raw_text", "")
    case_learning_contexts = state.get("case_learning_contexts", [])

    # Index case learning by state name for quick lookup
    cl_by_state = {}
    for clc in case_learning_contexts:
        cl_by_state[clc.get("state_name", "")] = clc

    decompositions = []
    all_variables = []

    completed_count = 0
    total = len(state_names)

    async def _process(state_name: str):
        nonlocal completed_count
        await sqlite_db.update_run_progress(
            state["run_id"],
            f"State Decomposition: generating {state_name}..."
        )
        case_learning = cl_by_state.get(state_name)

        user_prompt = _build_user_prompt(
            context_json, state_name, state_names, raw_text, case_learning
        )

        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=user_prompt),
        ]

        parsed = await json_ainvoke_with_retry(llm, messages)
        if isinstance(parsed, list):
            parsed = {"cases": parsed}
            
        parsed["state_name"] = state_name  # Ensure correct state name
        parsed = _coerce_decomposition(parsed)

        decomposition = StateDecomposition(**parsed)
        
        completed_count += 1
        await sqlite_db.update_run_progress(
            state["run_id"],
            f"State Decomposition {completed_count}/{total}: finished {state_name}"
        )
        return decomposition

    tasks = [_process(sn) for sn in state_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            logger.error(f"[Agent2] Error in async decomposition: {res}")
            continue

        decompositions.append(res.model_dump())

        # Aggregate variables
        for v in res.extracted_variables:
            v_name = v["name"] if isinstance(v, dict) else getattr(v, "name", None)
            if v_name and v_name not in [av["name"] for av in all_variables]:
                all_variables.append(v if isinstance(v, dict) else v.model_dump())

    # Also build legacy state_specs for backward compatibility
    state_specs = []
    for d in decompositions:
        # Flatten cases into expected_user_input / expected_bot_output
        user_inputs = []
        bot_outputs = []
        for c in d.get("cases", []):
            user_inputs.append(f"[{c['category']}] {c['description']}")
            bot_outputs.append(f"[{c['category']}] {c['handling_hint']}")

        state_specs.append({
            "state_name": d["state_name"],
            "intent": d.get("intent", ""),
            "expected_user_input": "\n".join(user_inputs),
            "expected_bot_output": "\n".join(bot_outputs),
            "dependencies": d.get("dependencies", []),
            "tags": d.get("tags", []),
        })

    logger.info(
        f"Agent 2: Decomposed {len(decompositions)} states, "
        f"extracted {len(all_variables)} unique variables"
    )

    return {
        "state_decompositions": decompositions,
        "state_specs": state_specs,
        "extracted_variables": all_variables,
        "progress": f"States decomposed ({len(decompositions)} states, {sum(len(d['cases']) for d in decompositions)} total cases)",
    }
