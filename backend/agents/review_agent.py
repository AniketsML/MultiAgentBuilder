"""Pipeline Review Agent — Per-Dimension Critic Scoring.

Autonomous quality gate that validates every pipeline stage output
against the original context document. Returns a CriticScorecard with
per-dimension scores and targeted improvement instructions that route
directly to the responsible Improver Agent (bypassing Master Agent).
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.claude_client import get_llm
from backend.utils.json_parser import extract_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Pipeline Review Agent. You are the autonomous quality gate in a 15-agent self-improving system.

Your job: After each pipeline stage completes, you receive the stage output and the ORIGINAL context document. You must validate whether the output is faithful, complete, and high-quality.

Return a JSON object with PER-DIMENSION scores:
{
  "stage": "<stage_name>",
  "dimensions": {
    "persona_consistency": 0-100,
    "edge_case_coverage": 0-100,
    "tone_alignment": 0-100,
    "transition_accuracy": 0-100,
    "character_limit_compliance": 0-100
  },
  "total": 0-100,
  "passed": true/false,
  "failed_dimensions": ["list of dimension names scoring below 70"],
  "targeted_instructions": {
    "dimension_name": "specific, actionable instruction to fix this dimension"
  }
}

SCORING RULES:
- Each dimension is scored 0-100 independently.
- "total" is the weighted average: persona(25%) + edge_case(25%) + tone(15%) + transition(20%) + char_limit(15%).
- A stage PASSES if total >= 75 AND no single dimension is below 50.
- "failed_dimensions" lists ONLY dimensions scoring below 70.
- "targeted_instructions" MUST have an entry for EVERY failed dimension. Be extremely specific — tell the agent exactly what to fix, not vague guidance.

DIMENSION DEFINITIONS:
- persona_consistency: Does the output match the bot's defined persona, voice, and character?
- edge_case_coverage: Does the output handle all edge cases, objections, refusals, and variations?
- tone_alignment: Is the emotional register consistent with the defined tone?
- transition_accuracy: Are state transitions, dependencies, and handoffs correctly specified?
- character_limit_compliance: Are all outputs within specified length limits?

RULES:
- Be strict but fair. Compare against the context document for factual accuracy.
- Do NOT hallucinate issues. Only flag real problems with specific evidence.
- Respond with ONLY the JSON object. No markdown, no preamble."""


async def review_stage_output(
    stage_name: str,
    stage_output: dict,
    context_doc: str,
    is_cold_start: bool = False,
) -> dict:
    """Review a single pipeline stage output. Returns CriticScorecard as dict."""
    llm = get_llm(max_tokens=2000)

    # On cold start, add context for the reviewer
    cold_start_note = ""
    if is_cold_start:
        cold_start_note = """
NOTE: This is a COLD START run — no KB references were available.
Lower your expectations for style matching (persona_consistency and tone_alignment)
but maintain strict standards for edge_case_coverage and transition_accuracy.
Passing threshold is 75 instead of 82 for cold start runs."""

    user_msg = f"""ORIGINAL CONTEXT DOCUMENT:
---
{context_doc}
---

PIPELINE STAGE: {stage_name}
STAGE OUTPUT:
{json.dumps(stage_output, indent=2, default=str)}
{cold_start_note}

Review this output for accuracy, completeness, and quality relative to the context document."""

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    parsed = extract_json(response.content)

    # Ensure required fields exist
    if "dimensions" not in parsed:
        parsed["dimensions"] = {
            "persona_consistency": 80,
            "edge_case_coverage": 80,
            "tone_alignment": 80,
            "transition_accuracy": 80,
            "character_limit_compliance": 100,
        }
    if "total" not in parsed:
        dims = parsed["dimensions"]
        parsed["total"] = int(
            dims.get("persona_consistency", 80) * 0.25
            + dims.get("edge_case_coverage", 80) * 0.25
            + dims.get("tone_alignment", 80) * 0.15
            + dims.get("transition_accuracy", 80) * 0.20
            + dims.get("character_limit_compliance", 100) * 0.15
        )
    if "passed" not in parsed:
        threshold = 75 if is_cold_start else 82
        min_dim = min(parsed["dimensions"].values()) if parsed["dimensions"] else 0
        parsed["passed"] = parsed["total"] >= threshold and min_dim >= 50
    if "failed_dimensions" not in parsed:
        parsed["failed_dimensions"] = [
            k for k, v in parsed.get("dimensions", {}).items() if v < 70
        ]
    if "targeted_instructions" not in parsed:
        parsed["targeted_instructions"] = {}
    if "stage" not in parsed:
        parsed["stage"] = stage_name

    logger.info(
        f"[ReviewAgent] {stage_name}: {parsed['total']}/100 | "
        f"passed={parsed['passed']} | failed={parsed.get('failed_dimensions', [])}"
    )
    return parsed


async def review_full_pipeline(pipeline_state: dict) -> list[dict]:
    """Review all completed stages of a pipeline run."""
    context_doc = pipeline_state.get("context_doc", "")
    is_cold_start = pipeline_state.get("is_cold_start", False)
    reviews = []

    stage_map = {
        "context_analysis": pipeline_state.get("context_schema"),
        "pattern_abstraction": pipeline_state.get("pattern_analysis"),
        "state_planning": pipeline_state.get("state_specs"),
        "variable_extraction": pipeline_state.get("extracted_variables"),
        "prompt_writing": pipeline_state.get("drafts"),
    }

    for stage_name, output in stage_map.items():
        if output:
            review = await review_stage_output(
                stage_name, output, context_doc, is_cold_start=is_cold_start
            )
            reviews.append(review)

    return reviews
