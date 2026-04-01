"""Agent 6 — Pattern Abstractor.

Runs after Agent 1 (Context Analyser) and before Agent 2 (State Planner).
Takes raw_examples + style_patterns from ContextSchema + KB references
and outputs a PatternAnalysis: template skeleton, core rules, anti-patterns,
and slot priority. Agent 3 uses this for structure-driven prompt generation.
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.json_parser import extract_json
from backend.utils.prompt_loader import load_prompt
from backend.utils.context_chunker import chunk_for_agent
from backend.models.schemas import PipelineState, PatternAnalysis
from backend.kb.retrieval_engine import smart_retrieve

logger = logging.getLogger(__name__)


def get_system_prompt() -> str:
    return load_prompt("agent6")


async def abstract_patterns(state: PipelineState) -> dict:
    """LangGraph node: extract structural patterns from context + examples."""
    context_schema = state.get("context_schema", {})
    raw_text = state.get("raw_text", "")

    if not context_schema:
        return {
            "pattern_analysis": PatternAnalysis().model_dump(),
            "progress": "Pattern abstraction skipped (no context schema)",
        }

    domain = context_schema.get("domain", "")
    raw_examples = context_schema.get("raw_examples", [])
    style_patterns = context_schema.get("style_patterns", [])
    persona = context_schema.get("persona", "")

    # Get relevant context chunk for this agent
    chunked_text = chunk_for_agent(raw_text, "agent6")

    # Retrieve KB examples for pattern learning
    retrieval_ctx = await smart_retrieve(
        query=f"{domain} prompt structure patterns",
        domain=domain,
        persona=persona,
        state_intent="structural patterns and templates",
        n_results=5,
    )

    llm = get_llm(max_tokens=2000)

    # Build user prompt
    examples_text = "\n---\n".join(raw_examples) if raw_examples else "NONE PROVIDED"
    patterns_text = "\n".join(f"- {p}" for p in style_patterns) if style_patterns else "NONE"

    kb_text = "NONE AVAILABLE"
    if retrieval_ctx.examples:
        kb_parts = []
        for i, (ex, score) in enumerate(zip(retrieval_ctx.examples, retrieval_ctx.scores), 1):
            kb_parts.append(f"KB Example {i} (confidence: {score:.2f}):\n{ex[:500]}")
        kb_text = "\n---\n".join(kb_parts)

    user_prompt = f"""Context document (relevant sections):
---
{chunked_text[:3000]}
---

User's sample prompts (gold standard):
---
{examples_text[:3000]}
---

Style patterns already identified:
{patterns_text}

KB reference examples:
---
{kb_text}
---

Retrieval note: {retrieval_ctx.retrieval_note}

Analyse ALL the above inputs and extract the structural patterns now."""

    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw_response = response.content
    parsed = extract_json(raw_response)

    # Validate and coerce
    if isinstance(parsed.get("core_rules"), str):
        parsed["core_rules"] = [parsed["core_rules"]]
    if isinstance(parsed.get("anti_patterns"), str):
        parsed["anti_patterns"] = [parsed["anti_patterns"]]
    if isinstance(parsed.get("slot_priority"), str):
        parsed["slot_priority"] = [parsed["slot_priority"]]

    pattern = PatternAnalysis(**parsed)

    logger.info(
        f"Agent 6: Extracted pattern with {len(pattern.core_rules)} rules, "
        f"{len(pattern.anti_patterns)} anti-patterns"
    )

    return {
        "pattern_analysis": pattern.model_dump(),
        "progress": "Patterns abstracted",
    }
