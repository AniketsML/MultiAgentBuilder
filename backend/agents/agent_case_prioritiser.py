"""Case Prioritiser — new agent.

Takes StateDecomposition and ranks cases by:
- occurrence_probability: how often will this actually happen?
- criticality: what's the cost of handling this badly?

Cases below threshold on both dimensions get filtered or merged.
High-criticality low-frequency cases (escalation) stay.
Low-criticality low-frequency cases get consolidated into catch-all.
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.json_parser import extract_json, json_ainvoke_with_retry
from backend.utils.prompt_loader import load_prompt
from backend.models.schemas import (
    PipelineState, PrioritisedCase, PrioritisedCaseList, VariableSchema,
)

logger = logging.getLogger(__name__)


def get_system_prompt() -> str:
    return load_prompt("case_prioritiser")


def _build_user_prompt(
    context_schema_json: str,
    decomposition: dict,
    char_limit: int,
) -> str:
    cases_json = json.dumps(decomposition.get("cases", []), indent=2)
    state_name = decomposition.get("state_name", "")
    intent = decomposition.get("intent", "")

    return f"""Context Schema:
{context_schema_json}

STATE: {state_name}
INTENT: {intent}
CHARACTER BUDGET: {char_limit} characters for the final assembled prompt

CASES TO PRIORITISE:
{cases_json}

For each case, score:
- occurrence_probability (0-100): how often will this case actually happen in production?
- criticality (0-100): what's the cost if the bot handles this case badly?
  - Legal/compliance risk = 90-100
  - User frustration/churn risk = 70-89
  - Suboptimal but recoverable = 40-69
  - Cosmetic/minor = 0-39

Then decide an action for each case:
- "keep": include as its own handler in the final prompt
- "merge": merge into another case's handler (specify merge_into)
- "filter": remove entirely (only for truly irrelevant cases)

RULES:
- High criticality (>70) cases ALWAYS get "keep", even if low probability
- happy_path ALWAYS gets "keep"
- escalation_trigger ALWAYS gets "keep"
- Cases with both probability <20 AND criticality <30 can be "filter" or "merge"
- When merging, merge low-priority into the most relevant higher-priority case
- priority_score = (0.4 * probability + 0.6 * criticality) for ranking

Return ONLY a JSON object:
{{
  "state_name": "{state_name}",
  "intent": "{intent}",
  "cases": [
    {{
      "case_name": "...",
      "category": "...",
      "description": "...",
      "handling_hint": "...",
      "required_variables": [...],
      "transition_to": "...",
      "tone_guidance": "...",
      "occurrence_probability": 0-100,
      "criticality": 0-100,
      "priority_score": 0-100,
      "action": "keep|merge|filter",
      "merge_into": ""
    }}
  ],
  "filtered_count": 0,
  "merged_count": 0
}}"""


def _coerce_prioritised_case(raw: dict) -> dict:
    """Normalize LLM output for PrioritisedCase."""
    for field in ("occurrence_probability", "criticality", "priority_score"):
        val = raw.get(field)
        if val is None:
            raw[field] = 50
        elif isinstance(val, str):
            try:
                raw[field] = int(val)
            except ValueError:
                raw[field] = 50

    raw.setdefault("action", "keep")
    raw.setdefault("merge_into", "")
    raw.setdefault("case_name", "")
    raw.setdefault("category", "happy_path")
    raw.setdefault("description", "")
    raw.setdefault("handling_hint", "")
    raw.setdefault("transition_to", "")
    raw.setdefault("tone_guidance", "")

    if raw.get("required_variables") is None:
        raw["required_variables"] = []
    elif isinstance(raw["required_variables"], str):
        raw["required_variables"] = [raw["required_variables"]] if raw["required_variables"].strip() else []

    return raw


async def prioritise_cases(state: PipelineState) -> dict:
    """LangGraph node: prioritise cases for all states."""
    llm = get_llm(max_tokens=4096)

    context_schema = state.get("context_schema", {})
    decompositions = state.get("state_decompositions", [])
    context_json = json.dumps(context_schema, indent=2)

    if not decompositions:
        return {
            "prioritised_cases": [],
            "progress": "Case prioritisation skipped (no decompositions)",
        }

    prioritised_lists = []

    for decomposition in decompositions:
        state_name = decomposition.get("state_name", "")

        # Determine char limit
        char_limit = 2000 if state_name in ("global_instructions", "global_instruction") else 4500

        user_prompt = _build_user_prompt(context_json, decomposition, char_limit)

        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=user_prompt),
        ]

        parsed = await json_ainvoke_with_retry(llm, messages)
        if isinstance(parsed, list):
            parsed = {"cases": parsed}

        # Coerce cases
        cases = []
        for c in parsed.get("cases", []):
            c = _coerce_prioritised_case(c)
            cases.append(PrioritisedCase(**c))

        # Sort by priority_score descending
        cases.sort(key=lambda c: c.priority_score, reverse=True)

        # Build PrioritisedCaseList
        pcl = PrioritisedCaseList(
            state_name=state_name,
            intent=decomposition.get("intent", ""),
            cases=cases,
            filtered_count=parsed.get("filtered_count", sum(1 for c in cases if c.action == "filter")),
            merged_count=parsed.get("merged_count", sum(1 for c in cases if c.action == "merge")),
            total_char_budget=char_limit,
            extracted_variables=[VariableSchema(**v) for v in decomposition.get("extracted_variables", [])],
            dependencies=decomposition.get("dependencies", []),
            tags=decomposition.get("tags", []),
        )
        prioritised_lists.append(pcl.model_dump())

    kept_total = sum(
        sum(1 for c in pcl["cases"] if c["action"] == "keep")
        for pcl in prioritised_lists
    )

    logger.info(
        f"Case Prioritiser: {len(prioritised_lists)} states, "
        f"{kept_total} cases kept"
    )

    return {
        "prioritised_cases": prioritised_lists,
        "progress": f"Cases prioritised ({kept_total} kept across {len(prioritised_lists)} states)",
    }
