"""KB Case Learner — replaces Agent 6 (Pattern Abstractor).

For each state, retrieves KB prompts and performs second-pass analysis:
- What categories of cases does each KB prompt handle?
- How does it handle happy path vs objections vs errors?
- What variables does it reference and in what conditions?
- What transition logic exists?
- What anti-patterns does it avoid?

Output: CaseLearningContext per state with structured case knowledge.
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.json_parser import extract_json
from backend.utils.prompt_loader import load_prompt
from backend.models.schemas import PipelineState, CaseLearningContext, CaseKnowledge
from backend.kb.retrieval_engine import retrieve_for_case_learning

logger = logging.getLogger(__name__)

# Canonical case taxonomy used across the system
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
    return load_prompt("kb_learner")


def _build_extraction_prompt(kb_prompt: str, domain: str, state_intent: str) -> str:
    """Build the second-pass extraction prompt for one KB prompt."""
    return f"""Analyse this KB prompt from the "{domain}" domain.
The state intent is: "{state_intent}"

KB PROMPT:
---
{kb_prompt[:3000]}
---

Extract structured case knowledge. For each case this prompt handles, identify:
1. case_category: one of [{', '.join(CASE_TAXONOMY)}]
2. handling_strategy: HOW does the prompt handle this case? (the approach, not the words)
3. variables_used: what data variables are referenced in this case handler?
4. transition_target: where does the conversation go after this case?
5. tone_approach: what tone/register is used for this case type?

Also identify:
- anti_patterns: things this prompt deliberately avoids doing

Return ONLY a JSON object:
{{
  "cases": [
    {{
      "case_category": "...",
      "handling_strategy": "...",
      "variables_used": ["..."],
      "transition_target": "...",
      "tone_approach": "..."
    }}
  ],
  "anti_patterns": ["..."]
}}"""


async def _extract_cases_from_prompt(
    kb_prompt: str, domain: str, state_intent: str
) -> tuple[list[dict], list[str]]:
    """Second-pass extraction: analyse one KB prompt for case knowledge."""
    llm = get_llm(max_tokens=1500)

    user_prompt = _build_extraction_prompt(kb_prompt, domain, state_intent)

    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
        parsed = extract_json(response.content)
        if isinstance(parsed, list):
            parsed = {"cases": parsed}

        cases = parsed.get("cases", [])
        anti_patterns = parsed.get("anti_patterns", [])

        return cases, anti_patterns
    except Exception as e:
        logger.warning(f"[KBLearner] Extraction failed for one prompt: {e}")
        return [], []


async def learn_cases_for_state(
    state_name: str,
    state_intent: str,
    domain: str,
    context_schema: dict,
) -> CaseLearningContext:
    """Learn case-handling strategies from KB for one state.

    Two-step retrieval:
    1. Semantic search for relevant KB prompts
    2. Dedicated extraction call per retrieved prompt
    """
    persona = context_schema.get("persona", "")
    guardrails = context_schema.get("guardrails", [])

    # Step 1: Retrieve relevant KB prompts with metadata
    candidates, is_cold_start = await retrieve_for_case_learning(
        domain=domain,
        state_name=state_name,
        state_intent=state_intent,
        case_categories=CASE_TAXONOMY[:6],  # Most common categories
        n_results=6,
    )

    if not candidates:
        return CaseLearningContext(
            state_name=state_name,
            source_count=0,
            learned_cases=[],
            common_variables=[],
            anti_patterns=[],
            retrieval_note="cold start -- no KB prompts found",
            is_cold_start=True,
        )

    # Step 2: Second-pass extraction for each retrieved prompt
    all_cases: list[CaseKnowledge] = []
    all_anti_patterns: list[str] = []
    all_variables: list[str] = []
    category_counts: dict[str, int] = {}

    for candidate in candidates:
        # If structured metadata already exists, use it directly
        if candidate.get("case_handling_map"):
            for cat, strategy in candidate["case_handling_map"].items():
                ck = CaseKnowledge(
                    case_category=cat,
                    handling_strategy=strategy,
                    variables_used=candidate.get("variables_used", []),
                    tone_approach="",
                )
                all_cases.append(ck)
                category_counts[cat] = category_counts.get(cat, 0) + 1
                all_variables.extend(candidate.get("variables_used", []))
        else:
            # Run LLM extraction on the raw prompt text
            cases_raw, anti_pats = await _extract_cases_from_prompt(
                candidate["prompt"], domain, state_intent
            )

            for c in cases_raw:
                cat = c.get("case_category", "")
                if cat not in CASE_TAXONOMY:
                    # Map to closest canonical category
                    cat = "happy_path"  # Default fallback

                ck = CaseKnowledge(
                    case_category=cat,
                    handling_strategy=c.get("handling_strategy", ""),
                    variables_used=c.get("variables_used", []),
                    transition_target=c.get("transition_target", ""),
                    tone_approach=c.get("tone_approach", ""),
                )
                all_cases.append(ck)
                category_counts[cat] = category_counts.get(cat, 0) + 1
                all_variables.extend(c.get("variables_used", []))

            all_anti_patterns.extend(anti_pats)

    # Deduplicate and find common variables
    common_variables = list(set(v for v in all_variables if all_variables.count(v) > 1))
    unique_anti_patterns = list(set(all_anti_patterns))

    retrieval_note = (
        f"Analysed {len(candidates)} KB prompts. "
        f"Found cases in categories: {', '.join(sorted(category_counts.keys()))}. "
        f"Most common: {max(category_counts, key=category_counts.get) if category_counts else 'none'}."
    )

    return CaseLearningContext(
        state_name=state_name,
        source_count=len(candidates),
        learned_cases=all_cases,
        common_variables=common_variables,
        anti_patterns=unique_anti_patterns,
        retrieval_note=retrieval_note,
        is_cold_start=is_cold_start,
    )


async def learn_cases(state: PipelineState) -> dict:
    """LangGraph node: learn case knowledge from KB for all states."""
    context_schema = state.get("context_schema", {})
    state_names = state.get("state_names", [])
    domain = context_schema.get("domain", "")

    if not context_schema or not state_names:
        return {
            "case_learning_contexts": [],
            "progress": "KB Case Learning skipped (no context/states)",
        }

    case_learning_contexts = []
    any_cold_start = False

    import asyncio
    completed = 0
    total = len(state_names)

    async def _process_state(state_name: str):
        nonlocal any_cold_start, completed
        state_intent = state_name.replace("_", " ")

        clc = await learn_cases_for_state(
            state_name=state_name,
            state_intent=state_intent,
            domain=domain,
            context_schema=context_schema,
        )
        completed += 1
        return clc

    tasks = [_process_state(sn) for sn in state_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            logger.error(f"[KBLearner] Error in async extraction: {res}")
            continue
            
        case_learning_contexts.append(res.model_dump())
        if res.is_cold_start:
            any_cold_start = True

    updates = {
        "case_learning_contexts": case_learning_contexts,
        "progress": f"KB Case Learning complete ({len(case_learning_contexts)} states analysed)",
    }

    if any_cold_start:
        updates["is_cold_start"] = True
        existing = list(state.get("cold_start_domains", []))
        if domain not in existing:
            existing.append(domain)
        updates["cold_start_domains"] = existing

    return updates
