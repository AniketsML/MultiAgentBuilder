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
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.json_parser import extract_json
from backend.utils.prompt_loader import load_prompt
from backend.models.schemas import (
    PipelineState, CaseHandler, CaseWriterOutput,
)

logger = logging.getLogger(__name__)


def get_system_prompt() -> str:
    return load_prompt("case_writer")


def _build_user_prompt(
    context_schema_json: str,
    prioritised_case_list: dict,
    case_learning: dict | None,
    variables_json: str,
    mixed_dna_principles: list[str] | None = None,
) -> str:
    state_name = prioritised_case_list.get("state_name", "")
    intent = prioritised_case_list.get("intent", "")
    char_budget = prioritised_case_list.get("total_char_budget", 4500)

    # Only include cases with action="keep"
    kept_cases = [c for c in prioritised_case_list.get("cases", []) if c.get("action") == "keep"]
    cases_json = json.dumps(kept_cases, indent=2)

    # Format KB case learning
    kb_section = ""
    if case_learning and case_learning.get("learned_cases"):
        strategies = []
        for lc in case_learning["learned_cases"]:
            strategies.append(
                f"  [{lc['case_category']}]: {lc['handling_strategy'][:150]}"
                f" | tone: {lc.get('tone_approach', 'N/A')}"
            )
        kb_section = f"""
LEARNED HANDLING STRATEGIES (from KB -- learn the approach, don't copy words):
{chr(10).join(strategies[:15])}
"""

    # Paradigm DNA constraints
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

LOCKED VARIABLES (use ONLY these, never invent new ones -- ALWAYS wrap in double curly braces {{{{var_name}}}}):
{variables_json}
{kb_section}{dna_section}

CASES TO WRITE HANDLERS FOR:
{cases_json}

For EACH case, write a self-contained handling block as a JSON object:
{{
  "case_name": "...",
  "category": "...",
  "condition": "What specific user input or system state triggers this handler",
  "bot_response": "The actual instruction text for the bot -- what it says and does for this case. Write it as a prompt instruction, not as dialogue.",
  "variables_used": ["only from the locked list"],
  "transition_to": "next state name or empty",
  "tone": "tone used in this handler",
  "char_count": 0
}}

RULES:
- Each handler must be SELF-CONTAINED -- readable without other handlers
- Match the persona from context_schema.persona in every handler
- Tone varies per case type:
  - objection_refusal: empathetic, acknowledge concern, offer alternative
  - wrong_format / repeat_loop: patient, clear, provide example
  - escalation_trigger: calm, de-escalation language
  - system_condition: clear, brief, solution-oriented
  - happy_path: match overall persona tone
- VARIABLE FORMAT (MANDATORY): Every variable reference MUST use double curly braces.
  Write {{{{first_name}}}} not [first_name] or first_name. No exceptions.
- Use ONLY locked variables -- never hallucinate {{{{unlisted_var}}}}
- bot_response should be an instruction to the bot, not a script
- Keep each handler tight -- you have a char budget to share across ALL handlers
- TRANSITION/ROUTING (MANDATORY): Every handler MUST end with explicit routing:
  - Simple transitions: "→ Transition to [next_state_name]"
  - Conditional routing: "If {{{{var}}}} indicates X → state_a, otherwise → state_b"
  - Routing must reference EXACT state names from the flow
  - If the handler loops back to the same state (e.g., retry), state that explicitly

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
    llm = get_llm(max_tokens=3500)

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

    for pcl in prioritised_cases:
        state_name = pcl.get("state_name", "")
        case_learning = cl_by_state.get(state_name)

        user_prompt = _build_user_prompt(
            context_json, pcl, case_learning, variables_json,
            mixed_dna_principles=dna_principles,
        )

        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=user_prompt),
        ]

        response = await llm.ainvoke(messages)
        parsed = extract_json(response.content)

        handlers = []
        for h in parsed.get("handlers", []):
            h = _coerce_handler(h)
            handlers.append(CaseHandler(**h))

        total_chars = sum(h.char_count for h in handlers)

        output = CaseWriterOutput(
            state_name=state_name,
            handlers=handlers,
            total_char_count=total_chars,
        )
        all_handlers.append(output.model_dump())

    total_handlers = sum(len(h["handlers"]) for h in all_handlers)

    logger.info(
        f"Case Writer: {len(all_handlers)} states, "
        f"{total_handlers} total handlers written"
    )

    return {
        "case_handlers": all_handlers,
        "progress": f"Case handlers written ({total_handlers} handlers across {len(all_handlers)} states)",
    }
