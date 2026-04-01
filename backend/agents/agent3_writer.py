"""Agent 3 — Prompt Writer node.

RAG-augmented via the advanced retrieval engine with pattern-driven generation.
Uses template_skeleton + core_rules from Pattern Abstractor for structure.
Cold start mode: falls back to pattern-only generation without style mimicry.
"""

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.models.schemas import PipelineState, PromptDraft, RetrievalContext
from backend.kb.retrieval_engine import smart_retrieve


from backend.utils.prompt_loader import load_prompt

def _get_system_prompt(domain: str) -> str:
    """Return the system prompt with domain injected."""
    template = load_prompt("agent3")
    return template.replace("{domain}", domain)


def _build_user_prompt(
    context_schema_json: str,
    state_spec_json: str,
    variables_json: str,
    retrieval_ctx: RetrievalContext | None = None,
    pattern_analysis: dict | None = None,
    regen_reason: str | None = None,
) -> str:
    # Build retrieval examples section with confidence scores
    if retrieval_ctx and retrieval_ctx.examples:
        example_parts = []
        for i, (ex, score) in enumerate(zip(retrieval_ctx.examples, retrieval_ctx.scores)):
            confidence = "HIGH" if score > 0.85 else "MODERATE" if score > 0.70 else "LOW"
            example_parts.append(
                f"Example {i+1} (confidence: {confidence}, score: {score:.2f}):\n{ex}"
            )
        examples_text = "\n---\n".join(example_parts)
        retrieval_note = retrieval_ctx.retrieval_note
    else:
        examples_text = "No examples available in the KB."
        retrieval_note = "cold start — no KB reference"

    # Build pattern analysis section
    pattern_section = ""
    if pattern_analysis and pattern_analysis.get("template_skeleton"):
        pattern_section = f"""
STRUCTURAL PATTERNS (from Pattern Abstractor):
Template skeleton:
{pattern_analysis.get('template_skeleton', 'N/A')}

Core rules to follow:
{chr(10).join(f'- {r}' for r in pattern_analysis.get('core_rules', []))}

Anti-patterns to avoid:
{chr(10).join(f'- {ap}' for ap in pattern_analysis.get('anti_patterns', []))}

Slot priority (domain-specific vs reusable):
{chr(10).join(f'- {s}' for s in pattern_analysis.get('slot_priority', []))}
"""

    # Cold start instructions
    cold_start_note = ""
    if retrieval_ctx and retrieval_ctx.is_cold_start:
        cold_start_note = """
COLD START MODE: No matching KB entries were found for this domain.
- Do NOT attempt to mimic a style you haven't seen.
- Focus on STRUCTURE-DRIVEN generation using the template skeleton and core rules above.
- Prioritize edge case coverage and transition accuracy over style matching.
- Write a clean, well-structured prompt that can serve as the seed for future KB entries."""

    prompt = f"""Context schema:
{context_schema_json}

Variables needed for this flow:
{variables_json}

State to write:
{state_spec_json}
{pattern_section}

Reference examples from the user's KB (study style, not content):
---
{examples_text}
---
Retrieval note: {retrieval_note}
{cold_start_note}

Write the prompt for this state now."""

    if regen_reason:
        prompt += f"""

IMPORTANT — REGENERATION NOTE:
The previous attempt at this prompt was rejected by the user.
Rejection reason: '{regen_reason}'

Do NOT repeat the mistake described above.
Write a new prompt that addresses this feedback while still following
all format rules and staying within the domain."""

    return prompt


async def write_prompts(state: PipelineState) -> dict:
    """LangGraph node: write prompts for all states with advanced retrieval."""
    context_schema = state["context_schema"]
    state_specs = state["state_specs"]
    extracted_vars = state.get("extracted_variables", [])
    pattern_analysis = state.get("pattern_analysis")
    domain = context_schema["domain"]
    persona = context_schema.get("persona", "")
    guardrails = context_schema.get("guardrails", [])
    escalation_triggers = context_schema.get("escalation_triggers", [])
    is_cold_start = state.get("is_cold_start", False)

    llm = get_llm(max_tokens=2500)
    system_prompt = _get_system_prompt(domain)

    drafts = []
    retrieval_contexts = []

    for idx, spec in enumerate(state_specs):
        # Advanced RAG retrieval per state
        retrieval_ctx = await smart_retrieve(
            query=f"{domain} {spec['state_name']} {spec['intent']}",
            domain=domain,
            state_intent=spec.get("intent", ""),
            persona=persona,
            guardrails=guardrails,
            escalation_triggers=escalation_triggers,
            tags=spec.get("tags", []),
            n_results=5,
        )

        # Track cold start at pipeline level
        if retrieval_ctx.is_cold_start and not is_cold_start:
            is_cold_start = True

        user_prompt = _build_user_prompt(
            json.dumps(context_schema, indent=2),
            json.dumps(spec, indent=2),
            json.dumps(extracted_vars, indent=2),
            retrieval_ctx=retrieval_ctx,
            pattern_analysis=pattern_analysis,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await llm.ainvoke(messages)
        prompt_text = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()

        drafts.append(
            PromptDraft(
                state_name=spec["state_name"],
                prompt=prompt_text,
                retrieved_examples=retrieval_ctx.examples,
            ).model_dump()
        )

        retrieval_contexts.append(retrieval_ctx.model_dump())

    return {
        "drafts": drafts,
        "retrieval_contexts": retrieval_contexts,
        "is_cold_start": is_cold_start,
        "progress": "All prompts written",
    }


async def regenerate_prompt(state: PipelineState) -> dict:
    """LangGraph node: regenerate a single prompt after discard."""
    context_schema = state["context_schema"]
    domain = context_schema["domain"]
    persona = context_schema.get("persona", "")
    guardrails = context_schema.get("guardrails", [])
    escalation_triggers = context_schema.get("escalation_triggers", [])
    extracted_vars = state.get("extracted_variables", [])
    pattern_analysis = state.get("pattern_analysis")
    regen_state_name = state.get("regen_state_name", "")
    regen_reason = state.get("regen_reason", "")

    if not regen_reason:
        regen_reason = (
            "The previous attempt did not meet the user's expectations. "
            "Try a different approach."
        )

    # Find the matching state spec
    target_spec = None
    for spec in state["state_specs"]:
        if spec["state_name"] == regen_state_name:
            target_spec = spec
            break

    if not target_spec:
        return {"error": f"State '{regen_state_name}' not found in specs"}

    llm = get_llm(max_tokens=2500)
    system_prompt = _get_system_prompt(domain)

    # Advanced RAG retrieval
    retrieval_ctx = await smart_retrieve(
        query=f"{domain} {target_spec['state_name']} {target_spec['intent']}",
        domain=domain,
        state_intent=target_spec.get("intent", ""),
        persona=persona,
        guardrails=guardrails,
        escalation_triggers=escalation_triggers,
        tags=target_spec.get("tags", []),
        n_results=5,
    )

    user_prompt = _build_user_prompt(
        json.dumps(context_schema, indent=2),
        json.dumps(target_spec, indent=2),
        json.dumps(extracted_vars, indent=2),
        retrieval_ctx=retrieval_ctx,
        pattern_analysis=pattern_analysis,
        regen_reason=regen_reason,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    new_prompt = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()

    # Update the specific draft in the list
    drafts = list(state.get("drafts", []))
    for i, d in enumerate(drafts):
        if d["state_name"] == regen_state_name:
            drafts[i] = PromptDraft(
                state_name=regen_state_name,
                prompt=new_prompt,
                retrieved_examples=retrieval_ctx.examples,
                status="pending",
            ).model_dump()
            break

    return {
        "drafts": drafts,
        "regen_state_name": None,
        "regen_reason": None,
        "progress": f"Regenerated prompt for {regen_state_name}",
    }
