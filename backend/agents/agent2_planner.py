"""Agent 2 — State Planner node.

Uses the verbatim system prompt from spec §4.2.
Returns list[StateSpec] as dicts in the pipeline state.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from backend.utils.json_parser import extract_json

from backend.agents.claude_client import get_llm
from backend.models.schemas import PipelineState, StateSpec

from backend.utils.prompt_loader import load_prompt

def get_system_prompt() -> str:
    return load_prompt("agent2")


def _build_user_prompt(context_schema_json: str, state_names: list[str], raw_text: str) -> str:
    names_str = "\n".join(f"- {name}" for name in state_names)
    return f"""Original Context Document:
---
{raw_text}
---

Extracted Context Schema:
{context_schema_json}

State names to plan:
{names_str}

Produce the StateSpec for each state now."""


def _coerce_state_spec(raw: dict) -> dict:
    """Normalize LLM output: join lists to strings, ensure list fields are lists."""
    # Fields that should be strings — join if LLM returned a list
    for field in ("expected_user_input", "expected_bot_output", "intent"):
        val = raw.get(field)
        if isinstance(val, list):
            raw[field] = "\n".join(str(v) for v in val)
        elif val is None:
            raw[field] = ""

    # Fields that should be lists — wrap if LLM returned a string
    for field in ("dependencies", "tags"):
        val = raw.get(field)
        if val is None:
            raw[field] = []
        elif isinstance(val, str):
            raw[field] = [val] if val.strip() else []

    return raw


async def plan_states(state: PipelineState) -> dict:
    """LangGraph node: plan all states with intents and dependencies."""
    llm = get_llm(max_tokens=3000)

    context_json = json.dumps(state["context_schema"], indent=2)
    user_prompt = _build_user_prompt(context_json, state["state_names"], state.get("raw_text", ""))

    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw_text = response.content
    parsed = extract_json(raw_text)

    # Coerce and validate each spec
    state_specs = [StateSpec(**_coerce_state_spec(s)).model_dump() for s in parsed]

    return {
        "state_specs": state_specs,
        "progress": "States planned",
    }

