"""Pipeline Review Agent — Per-Gate Specialized Rubrics.

Each gate has its own rubric instead of one universal rubric.
Cross-reference check: decomposition vs final prompt.
Per-case-handler quality floor.
Tone-per-case-type enforcement.
Cross-agent causation detection for targeted improvement routing.
Paradigm compliance scoring for DNA-aware stages.
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from backend.agents.claude_client import get_llm
from backend.utils.json_parser import extract_json

logger = logging.getLogger(__name__)

# ─────────────── Per-Gate Rubric Definitions ───────────────

GATE_RUBRICS = {
    "kb_case_learning": {
        "system_addendum": """You are reviewing the KB Case Learner output.

SCORING DIMENSIONS FOR THIS GATE:
- extraction_completeness (40%): Did it find ALL case categories present in the retrieved KB prompts?
  Coverage across happy/objection/error/escalation buckets.
- variable_discovery (20%): Were variables correctly identified from case contexts?
- strategy_quality (25%): Are the extracted handling strategies specific and actionable,
  not vague summaries?
- anti_pattern_identification (15%): Were real anti-patterns identified, not generic ones?

Do NOT score character_limit_compliance or transition_accuracy -- they don't apply here.""",
        "dimensions": ["extraction_completeness", "variable_discovery", "strategy_quality", "anti_pattern_identification"],
        "weights": {"extraction_completeness": 0.40, "variable_discovery": 0.20, "strategy_quality": 0.25, "anti_pattern_identification": 0.15},
    },

    "state_decomposition": {
        "system_addendum": """You are reviewing the State Decomposer output.

SCORING DIMENSIONS FOR THIS GATE:
- taxonomy_completeness (35%): Are all 12 canonical case categories considered?
  Not all need to be included, but they should be explicitly considered.
- variable_extraction (20%): Do variables emerge naturally from cases? Are they specific?
- case_specificity (25%): Are case descriptions specific to this domain, not generic?
- dependency_accuracy (20%): Are state dependencies correct and complete?

Do NOT score character_limit_compliance -- it doesn't apply here.""",
        "dimensions": ["taxonomy_completeness", "variable_extraction", "case_specificity", "dependency_accuracy"],
        "weights": {"taxonomy_completeness": 0.35, "variable_extraction": 0.20, "case_specificity": 0.25, "dependency_accuracy": 0.20},
    },

    "case_prioritisation": {
        "system_addendum": """You are reviewing the Case Prioritiser output.

SCORING DIMENSIONS FOR THIS GATE:
- priority_accuracy (40%): Are high-criticality cases kept even if low-frequency?
  Are scores reasonable for this domain?
- filter_safety (30%): Are any obvious cases being dropped that should be kept?
  Escalation and happy_path must NEVER be filtered.
- merge_quality (20%): When cases are merged, is the merge-into target sensible?
- budget_awareness (10%): Does the kept case count fit within the char budget?

Do NOT score persona_consistency or tone_alignment -- they don't apply here.""",
        "dimensions": ["priority_accuracy", "filter_safety", "merge_quality", "budget_awareness"],
        "weights": {"priority_accuracy": 0.40, "filter_safety": 0.30, "merge_quality": 0.20, "budget_awareness": 0.10},
    },

    "case_writing": {
        "system_addendum": """You are reviewing the Case Writer output.

SCORING DIMENSIONS FOR THIS GATE -- SCORE EACH HANDLER INDIVIDUALLY:
- handler_quality (20%): Is each handler specific, actionable, and self-contained?
- tone_per_case (15%): Is the tone appropriate for each case TYPE?
  - objection cases need empathy
  - error/system cases need clarity and brevity
  - escalation cases need de-escalation language
  - repeat loop cases need patience without condescension
- variable_format_compliance (15%): Do ALL variable references use double curly braces?
  {{first_name}} is correct. [first_name] or bare first_name is WRONG.
  Score 0 if ANY handler uses non-compliant variable format.
- transition_routing (15%): Does every handler end with explicit routing?
  Each handler must state where the conversation goes next.
  Conditional routing ("if X → state_a, else → state_b") is expected for branching cases.
- variable_lock_compliance (10%): Does each handler use ONLY locked variables?
- persona_consistency (10%): Does the persona stay consistent across all handlers?
- paradigm_compliance (15%): If DNA principles were provided, does each handler
  demonstrably follow them? Check structural ordering, linguistic style,
  behavioral philosophy, recovery approach, and rhythm.

CRITICAL: Score EACH handler individually in per_case_scores. A single weak handler
for escalation_trigger should FAIL the gate even if other handlers are excellent.""",
        "dimensions": ["handler_quality", "tone_per_case", "variable_format_compliance", "transition_routing", "variable_lock_compliance", "persona_consistency", "paradigm_compliance"],
        "weights": {"handler_quality": 0.20, "tone_per_case": 0.15, "variable_format_compliance": 0.15, "transition_routing": 0.15, "variable_lock_compliance": 0.10, "persona_consistency": 0.10, "paradigm_compliance": 0.15},
    },

    "prompt_assembly": {
        "system_addendum": """You are reviewing the Prompt Assembler output.

SCORING DIMENSIONS FOR THIS GATE:
- coherence (15%): Does the assembled prompt read as one unified document, not a patchwork?
- case_coverage (15%): For EVERY case in the prioritised list, is there a corresponding
  handler in the final prompt? Cross-reference the case list against the assembled text.
- variable_format_compliance (15%): Do ALL variable references in the assembled prompt use
  double curly braces? {{first_name}} is correct. [first_name] or bare text is WRONG.
  Score 0 if ANY variable uses non-compliant format. This is non-negotiable.
- routing_section (15%): Does the prompt end with a ## ROUTING section that lists
  ALL transition paths from this state? Each path should show:
  condition → target_state_name. Missing or incomplete routing = score 0.
- character_limit_compliance (10%): Is the prompt within the char limit WITHOUT
  sacrificing case coverage?
- ordering_logic (10%): Is the case ordering logical? (happy path first, escalation last)
- transition_flow (10%): Are inline transitions between cases smooth and explicit?
- paradigm_compliance (10%): If DNA principles were embedded, does the assembled prompt
  follow the structural, linguistic, behavioral, and rhythm patterns specified?

You have access to BOTH the PrioritisedCaseList AND the assembled prompt.
Explicitly check: for every case with action="keep", is it present in the prompt?""",
        "dimensions": ["coherence", "case_coverage", "variable_format_compliance", "routing_section", "character_limit_compliance", "ordering_logic", "transition_flow", "paradigm_compliance"],
        "weights": {"coherence": 0.15, "case_coverage": 0.15, "variable_format_compliance": 0.15, "routing_section": 0.15, "character_limit_compliance": 0.10, "ordering_logic": 0.10, "transition_flow": 0.10, "paradigm_compliance": 0.10},
    },

    "consistency_review": {
        "system_addendum": """You are reviewing the cross-state consistency check output.

SCORING DIMENSIONS FOR THIS GATE:
- escalation_consistency (20%): Do all escalation triggers lead to the same destination?
- persona_consistency (20%): Is the persona identical across all states?
- case_taxonomy_alignment (20%): Are same case types handled consistently across states?
- overlap_detection (15%): No two states should handle the same case.
- handoff_completeness (10%): Are transitions between states complete with no dead ends?
- paradigm_consistency (15%): If DNA principles were applied, are they consistent
  across ALL states? Same structural pattern, same linguistic register, same
  behavioral philosophy throughout.""",
        "dimensions": ["escalation_consistency", "persona_consistency", "case_taxonomy_alignment", "overlap_detection", "handoff_completeness", "paradigm_consistency"],
        "weights": {"escalation_consistency": 0.20, "persona_consistency": 0.20, "case_taxonomy_alignment": 0.20, "overlap_detection": 0.15, "handoff_completeness": 0.10, "paradigm_consistency": 0.15},
    },

    "paradigm_mixing": {
        "system_addendum": """You are reviewing the Paradigm Mixer output.

SCORING DIMENSIONS FOR THIS GATE:
- source_diversity (25%): Were paradigms sourced from multiple prompts when available,
  not all from one source? Cross-domain sourcing is a positive signal.
- relevance_scoring (25%): Are the selected paradigms actually relevant to the target
  domain and use case? High-confidence selections preferred.
- conflict_detection (25%): Were genuine conflicts between paradigms detected and resolved
  with clear reasoning? Missing obvious conflicts is a failure.
- principle_quality (25%): Are the selected principles specific and actionable,
  not vague or generic?""",
        "dimensions": ["source_diversity", "relevance_scoring", "conflict_detection", "principle_quality"],
        "weights": {"source_diversity": 0.25, "relevance_scoring": 0.25, "conflict_detection": 0.25, "principle_quality": 0.25},
    },

    # Fallback for legacy stage names
    "context_analysis": {
        "system_addendum": "You are reviewing the Context Analyser output. Score on persona_consistency, edge_case_coverage, tone_alignment, transition_accuracy, character_limit_compliance using standard rubric.",
        "dimensions": ["persona_consistency", "edge_case_coverage", "tone_alignment", "transition_accuracy", "character_limit_compliance"],
        "weights": {"persona_consistency": 0.25, "edge_case_coverage": 0.25, "tone_alignment": 0.15, "transition_accuracy": 0.20, "character_limit_compliance": 0.15},
    },
}


BASE_SYSTEM_PROMPT = """You are the Pipeline Review Agent. You are the autonomous quality gate in a self-improving multi-agent system.

Your job: After each pipeline stage completes, you receive the stage output and the ORIGINAL context document. You must validate whether the output is faithful, complete, and high-quality.

{gate_rubric}

Return a JSON object with per-dimension scores:
{{
  "stage": "<stage_name>",
  "gate_type": "<gate_type>",
  "dimensions": {{
    "<dimension_1>": 0-100,
    "<dimension_2>": 0-100,
    ...
  }},
  "total": 0-100,
  "passed": true/false,
  "failed_dimensions": ["list of dimension names scoring below 70"],
  "targeted_instructions": {{
    "dimension_name": "specific, actionable instruction to fix this dimension"
  }},
  "per_case_scores": {{}},
  "root_cause_agent": ""
}}

SCORING RULES:
- Each dimension is scored 0-100 independently.
- "total" is the weighted average using the weights specified above.
- A stage PASSES if total >= 75 AND no single dimension is below 50.
- "targeted_instructions" MUST have an entry for EVERY failed dimension. Be extremely specific.
- "per_case_scores" should map case_name -> score (0-100) when reviewing case_writing stage.
- "root_cause_agent" should be set ONLY if the failure is caused by an UPSTREAM agent, not the agent being scored. Leave empty if the scored agent is at fault.

RULES:
- Be strict but fair. Compare against the context document for factual accuracy.
- Do NOT hallucinate issues. Only flag real problems with specific evidence.
- Respond with ONLY the JSON object. No markdown, no preamble."""


async def review_stage_output(
    stage_name: str,
    stage_output,
    context_doc: str,
    is_cold_start: bool = False,
    upstream_scorecards: list[dict] | None = None,
    prioritised_cases: list[dict] | None = None,
) -> dict:
    """Review a single pipeline stage output with gate-specific rubric."""
    llm = get_llm(max_tokens=2000)

    # Select gate-specific rubric
    rubric = GATE_RUBRICS.get(stage_name, GATE_RUBRICS.get("context_analysis"))
    gate_rubric = rubric["system_addendum"]
    dimensions = rubric["dimensions"]
    weights = rubric["weights"]

    system_prompt = BASE_SYSTEM_PROMPT.format(gate_rubric=gate_rubric)

    cold_start_note = ""
    if is_cold_start:
        cold_start_note = """
NOTE: This is a COLD START run -- no KB references were available.
Lower expectations for style matching but maintain strict standards for
case coverage and structural quality. Passing threshold is 75 instead of 82."""

    # Build cross-reference context for assembly gate
    cross_ref_section = ""
    if stage_name == "prompt_assembly" and prioritised_cases:
        case_summary = []
        for pcl in prioritised_cases:
            kept = [c["case_name"] for c in pcl.get("cases", []) if c.get("action") == "keep"]
            case_summary.append(f"  {pcl.get('state_name', '?')}: {', '.join(kept)}")
        cross_ref_section = f"""
EXPECTED CASES (from PrioritisedCaseList -- every "keep" case must appear in the prompt):
{chr(10).join(case_summary)}
"""

    # Cross-agent causation context
    upstream_section = ""
    if upstream_scorecards:
        failed_upstream = []
        for sc in upstream_scorecards:
            if not sc.get("passed", True):
                failed_upstream.append(
                    f"  {sc.get('stage', '?')}: failed on {', '.join(sc.get('failed_dimensions', []))}"
                )
        if failed_upstream:
            upstream_section = f"""
UPSTREAM FAILURES (check if this failure is caused by upstream, not this agent):
{chr(10).join(failed_upstream)}
If this stage's failure is caused by bad upstream input, set root_cause_agent to the
upstream agent ID (e.g., "kb_learner", "agent2", "case_prioritiser", "case_writer").
"""

    user_msg = f"""ORIGINAL CONTEXT DOCUMENT:
---
{context_doc[:3000]}
---

PIPELINE STAGE: {stage_name}
STAGE OUTPUT:
{json.dumps(stage_output, indent=2, default=str)[:4000]}
{cross_ref_section}{upstream_section}{cold_start_note}

Review this output now."""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ])

    parsed = extract_json(response.content)
    if isinstance(parsed, list):
        parsed = {}

    # Ensure required fields
    if "dimensions" not in parsed:
        parsed["dimensions"] = {d: 80 for d in dimensions}
    if "total" not in parsed:
        total = 0
        for dim, weight in weights.items():
            total += parsed["dimensions"].get(dim, 80) * weight
        parsed["total"] = int(total)
    if "passed" not in parsed:
        threshold = 75 if is_cold_start else 82
        dim_values = parsed["dimensions"].values()
        min_dim = min(dim_values) if dim_values else 0
        parsed["passed"] = parsed["total"] >= threshold and min_dim >= 50
    if "failed_dimensions" not in parsed:
        parsed["failed_dimensions"] = [
            k for k, v in parsed.get("dimensions", {}).items() if v < 70
        ]
    parsed.setdefault("targeted_instructions", {})
    parsed.setdefault("stage", stage_name)
    parsed.setdefault("gate_type", stage_name)
    parsed.setdefault("per_case_scores", {})
    parsed.setdefault("root_cause_agent", "")

    # Per-case quality floor check for case_writing gate
    if stage_name == "case_writing" and parsed.get("per_case_scores"):
        for case_name, score in parsed["per_case_scores"].items():
            if score < 50:
                parsed["passed"] = False
                if "handler_quality" not in parsed["failed_dimensions"]:
                    parsed["failed_dimensions"].append("handler_quality")
                parsed["targeted_instructions"][f"case_{case_name}"] = (
                    f"Handler for '{case_name}' scored {score}/100 -- below the 50-point floor. "
                    f"This handler must be rewritten."
                )

    logger.info(
        f"[ReviewAgent] {stage_name}: {parsed['total']}/100 | "
        f"passed={parsed['passed']} | failed={parsed.get('failed_dimensions', [])}"
        f"{' | root_cause=' + parsed['root_cause_agent'] if parsed.get('root_cause_agent') else ''}"
    )
    return parsed
