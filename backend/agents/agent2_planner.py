"""Agent 2 — State Decomposer (heavily upgraded from State Planner).

Takes one state at a time + CaseLearningContext from KB Case Learner.
Breaks the state into an exhaustive case map using the canonical taxonomy.
Variables are extracted here as a natural by-product of case decomposition.
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from backend.utils.json_parser import extract_json
from backend.agents.claude_client import get_llm
from backend.models.schemas import (
    PipelineState, StateDecomposition, CaseSpec, VariableSchema,
)
from backend.utils.prompt_loader import load_prompt

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


def _build_user_prompt(
    context_schema_json: str,
    state_name: str,
    state_names: list[str],
    raw_text: str,
    case_learning: dict | None = None,
) -> str:
    """Build the decomposition prompt for one state."""
    names_str = "\n".join(f"- {name}" for name in state_names)

    # Format KB case knowledge if available
    kb_section = ""
    if case_learning and case_learning.get("learned_cases"):
        kb_cases = []
        for lc in case_learning["learned_cases"]:
            kb_cases.append(
                f"  - {lc['case_category']}: {lc['handling_strategy'][:200]}"
                f" (vars: {', '.join(lc.get('variables_used', []))})"
            )
        kb_section = f"""
KB CASE KNOWLEDGE (from similar prompts in the knowledge base):
The following case categories have been seen for similar states:
{chr(10).join(kb_cases)}

Anti-patterns to avoid:
{chr(10).join(f'  - {ap}' for ap in case_learning.get('anti_patterns', []))}

Use this knowledge to inform which cases are most relevant for this domain.
Do NOT copy the strategies verbatim -- learn the reasoning approach.
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

CANONICAL CASE TAXONOMY (consider ALL of these):
{taxonomy_str}

Decompose this state into an exhaustive case map.
For EACH case, provide:
- case_name: descriptive snake_case name
- category: one of the canonical categories above
- description: what triggers this case (be specific)
- handling_hint: suggested handling approach
- required_variables: data variables needed for this case
- transition_to: which state comes next (from the state list)
- tone_guidance: appropriate tone for this case type

VARIABLE FORMAT CONVENTION (MANDATORY):
- All variable names MUST be written in double curly braces: {{{{variable_name}}}}
- Examples: {{{{first_name}}}}, {{{{payment_amount}}}}, {{{{account_number}}}}
- Use snake_case inside the braces
- In the extracted_variables list, store the name WITHOUT braces (e.g. "first_name")
  but in handling_hint text, ALWAYS wrap them as {{{{first_name}}}}

TRANSITION LOGIC:
- transition_to must reference an EXACT state name from the state list above
- If the transition is conditional (different outcomes go to different states),
  describe the routing in the handling_hint, e.g.:
  "If {{{{user_response}}}} is affirmative → collect_payment, if refusal → handle_objection"

Also extract ALL variables that emerge from the cases.

Respond with ONLY a JSON object:
{{
  "state_name": "{state_name}",
  "intent": "one sentence describing this state's goal",
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
    llm = get_llm(max_tokens=3500)

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

    for state_name in state_names:
        case_learning = cl_by_state.get(state_name)

        user_prompt = _build_user_prompt(
            context_json, state_name, state_names, raw_text, case_learning
        )

        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=user_prompt),
        ]

        response = await llm.ainvoke(messages)
        raw_text_resp = response.content
        parsed = extract_json(raw_text_resp)
        parsed["state_name"] = state_name  # Ensure correct state name
        parsed = _coerce_decomposition(parsed)

        decomposition = StateDecomposition(**parsed)
        decompositions.append(decomposition.model_dump())

        # Aggregate variables
        for v in decomposition.extracted_variables:
            if v.name not in [av["name"] for av in all_variables]:
                all_variables.append(v.model_dump())

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
