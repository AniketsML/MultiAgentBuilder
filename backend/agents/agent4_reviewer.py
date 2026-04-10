"""Agent 4 — Consistency Reviewer (updated for case-based pipeline + Prompt DNA).

Now also checks:
- Case taxonomy consistency across states
- Escalation triggers lead to the same destination across all states
- No overlapping case responsibilities between states
- Case handler coverage matches the prioritised case list
- Paradigm consistency: all states comply with the MixedDNA principles
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from backend.utils.json_parser import extract_json
from backend.agents.claude_client import get_llm
from backend.models.schemas import PipelineState, ReviewResult
from backend.utils.prompt_loader import load_prompt


def get_system_prompt() -> str:
    return load_prompt("agent4")


def _build_user_prompt(
    context_schema_json: str,
    all_drafts_json: str,
    prioritised_cases_json: str = "[]",
    dna_principles: list[str] | None = None,
) -> str:
    dna_section = ""
    if dna_principles:
        dna_lines = "\n".join(f"  - {p}" for p in dna_principles[:20])
        dna_section = f"""

10. PARADIGM COMPLIANCE: The following architectural principles were learned from the KB.
    Check that ALL prompts comply with them consistently:
{dna_lines}

    Flag any prompt that violates a principle as a consistency issue.
"""
    return f"""Context schema:
{context_schema_json}

All draft prompts:
{all_drafts_json}

Prioritised case lists (what SHOULD be in each prompt):
{prioritised_cases_json}

REVIEW CHECKLIST:
1. PERSONA DRIFT: Does the bot's voice change across states?
2. CONTRADICTIONS: Does one state contradict instructions in another?
3. HANDOFF GAPS: Are transitions between dependent states clean?
4. TONE INCONSISTENCY: Are formality levels consistent across the persona?
5. CASE TAXONOMY CONSISTENCY: If state A handles "user_refuses" one way and state B handles
   it differently, flag the inconsistency (unless the difference is justified by context).
6. ESCALATION CONSISTENCY: Do all escalation triggers across all states lead to the same
   destination? If not, flag it.
7. OVERLAP: Do two states handle the same case? Each case belongs to exactly one state.
8. CASE COVERAGE: For each state, check that every case in the prioritised list with
   action="keep" is actually handled in the assembled prompt.
9. MISSING EDGE CASES: Does any state fail to handle a case it should own?
10. VARIABLE FORMAT COMPLIANCE: ALL variable references in EVERY prompt MUST use double
    curly braces format: {{first_name}}, {{payment_amount}}, etc.
    Flag ANY variable that uses [brackets], bare text, or any other format.
    This is a HARD REQUIREMENT -- any non-compliant variable is a finding.
11. ROUTING CONSISTENCY: Each prompt should have a ROUTING section at the end.
    Verify that:
    - Every prompt has explicit transition paths defined
    - Routing targets reference real state names from the flow
    - State A's routing to State B is consistent with State B's expectations
    - No dead-end states (every state must route somewhere)
    - Circular loops are explicitly marked as intentional (e.g., retry loops)
{dna_section}
Review for consistency now."""


async def review_consistency(state: PipelineState) -> dict:
    """LangGraph node: review all drafts for cross-prompt and case-level consistency."""
    llm = get_llm(max_tokens=2500)

    context_json = json.dumps(state["context_schema"], indent=2)

    # Format drafts for review
    drafts_for_review = []
    for d in state.get("drafts", []):
        drafts_for_review.append({
            "state_name": d["state_name"],
            "prompt": d["prompt"],
            "case_breakdown": d.get("case_breakdown", []),
        })
    drafts_json = json.dumps(drafts_for_review, indent=2)

    # Include prioritised cases for coverage check
    prioritised = state.get("prioritised_cases", [])
    pcl_summary = []
    for pcl in prioritised:
        kept = [c["case_name"] for c in pcl.get("cases", []) if c.get("action") == "keep"]
        pcl_summary.append({
            "state_name": pcl.get("state_name", ""),
            "expected_cases": kept,
        })
    pcl_json = json.dumps(pcl_summary, indent=2)

    # Extract MixedDNA principles for paradigm consistency check
    mixed_dna = state.get("mixed_dna")
    dna_principles = None
    if mixed_dna and not mixed_dna.get("is_cold_start", True):
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

    user_prompt = _build_user_prompt(
        context_json, drafts_json, pcl_json, dna_principles=dna_principles
    )

    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    raw_text = response.content
    parsed = extract_json(raw_text)
    review = ReviewResult(**parsed)

    return {
        "review_notes": review.overall_note,
        "review_findings": [f.model_dump() for f in review.findings],
        "progress": "Consistency review complete",
    }
