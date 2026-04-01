"""Agent 1 — Context & Style Analyser node.

RAG-augmented via the advanced retrieval engine: multi-signal queries,
BM25 hybrid search, MMR diversity, and cold-start handling.
Uses context chunking for large documents.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from backend.utils.json_parser import extract_json
from backend.agents.claude_client import get_llm
from backend.models.schemas import PipelineState, ContextSchema
from backend.utils.prompt_loader import load_prompt
from backend.utils.context_chunker import chunk_for_agent
from backend.kb.retrieval_engine import smart_retrieve


def get_system_prompt() -> str:
    return load_prompt("agent1")


async def _fetch_kb_references(context_doc: str, domain_hint: str = "") -> tuple:
    """Query KB via advanced retrieval engine. Returns (text, RetrievalContext)."""
    retrieval_ctx = await smart_retrieve(
        query=context_doc[:2000],
        domain=domain_hint,
        state_intent="context analysis and style extraction",
        n_results=5,
    )

    if not retrieval_ctx.examples:
        return "NONE FOUND", retrieval_ctx

    references = []
    for i, (ex, score) in enumerate(zip(retrieval_ctx.examples, retrieval_ctx.scores), 1):
        references.append(
            f"--- KB Example {i} (confidence: {score:.2f}) ---\n"
            f"{ex[:600]}\n"
        )
    return "\n".join(references), retrieval_ctx


def _build_user_prompt(
    context_doc: str,
    past_prompts: str | None,
    kb_refs: str,
    retrieval_ctx=None,
) -> str:
    past = past_prompts if past_prompts else "NONE"

    retrieval_note = ""
    if retrieval_ctx:
        retrieval_note = f"\nRetrieval confidence: {retrieval_ctx.retrieval_note}"
        if retrieval_ctx.is_cold_start:
            retrieval_note += "\nNOTE: This is a COLD START — no matching KB entries found. Rely entirely on the context document and sample prompts. Do not reference KB patterns you haven't seen."

    return f"""Context document:
---
{context_doc}
---

Sample prompts provided by user (gold standard for style/format):
---
{past}
---

Domain reference examples from Knowledge Base (use these to understand what guardrails, escalation triggers, error handling, and data requirements look like for similar domains — do NOT copy content, only learn what fields to extract):
---
{kb_refs}
---
{retrieval_note}

Extract the structured ContextSchema now."""


def _coerce_list_fields(parsed: dict) -> dict:
    """Normalize LLM output: convert dicts to list of values, strings to single-item lists."""
    list_fields = [
        "format_rules", "style_patterns", "raw_examples", "guardrails",
        "escalation_triggers", "error_handling", "required_data", "transition_rules",
    ]
    for field in list_fields:
        val = parsed.get(field)
        if val is None:
            parsed[field] = []
        elif isinstance(val, dict):
            parsed[field] = list(val.values())
        elif isinstance(val, str):
            parsed[field] = [val] if val.strip() else []
    return parsed


async def analyse_context(state: PipelineState) -> dict:
    """LangGraph node: analyse context document and extract schema."""
    llm = get_llm(max_tokens=2000)

    raw_text = state.get("raw_text", state.get("context_doc", ""))

    # Context chunking for large documents
    chunked_text = chunk_for_agent(raw_text, "agent1")

    # RAG: pull similar domain prompts from KB via advanced retrieval
    kb_refs, retrieval_ctx = await _fetch_kb_references(chunked_text)

    user_prompt = _build_user_prompt(
        chunked_text, state.get("past_prompts"), kb_refs, retrieval_ctx
    )

    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw_text_resp = response.content
    parsed = extract_json(raw_text_resp)
    parsed = _coerce_list_fields(parsed)
    context_schema = ContextSchema(**parsed)

    # Track cold start at pipeline level
    updates = {
        "context_schema": context_schema.model_dump(),
        "progress": "Context analysed",
    }

    if retrieval_ctx.is_cold_start:
        updates["is_cold_start"] = True
        domain = context_schema.domain
        existing = list(state.get("cold_start_domains", []))
        if domain not in existing:
            existing.append(domain)
        updates["cold_start_domains"] = existing

    return updates
