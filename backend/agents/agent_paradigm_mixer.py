"""Paradigm Mixer — blends best DNA per paradigm across the KB.

Runs AFTER KB Case Learner in the generation pipeline.
Queries the dna_kb for all available DNA records, scores each per-paradigm
for relevance to the current use case, selects the best DNA per paradigm
(can come from DIFFERENT source prompts), detects conflicts, and produces
a MixedDNA object.

This is the core capability: structural DNA from a banking prompt,
linguistic DNA from a healthcare prompt, behavioral DNA from an insurance
prompt — if those are the best available examples of each.
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from backend.agents.claude_client import get_llm
from backend.utils.json_parser import extract_json
from backend.models.schemas import (
    PipelineState, MixedDNA, ParadigmPrinciples,
    ParadigmConflict, PARADIGM_NAMES,
)
from backend.kb import sqlite_db

logger = logging.getLogger(__name__)


MIXER_SYSTEM = """You are the Paradigm Mixer — you select and blend the best prompt DNA across a knowledge base.

You will receive:
1. The TARGET domain and use case description
2. Multiple DNA records, each containing principles for 8 paradigms extracted from different source prompts

Your job:
1. For EACH of the 8 paradigms INDEPENDENTLY, score every available DNA source for relevance (0-100)
2. Select the BEST scoring source for each paradigm — they CAN come from different sources
3. Detect CONFLICTS between selected paradigms (e.g. structural says "brief recovery" but rhythm says "verbose always")
4. Resolve conflicts with a clear priority winner

SCORING CRITERIA per paradigm:
- Relevance: How well do these principles apply to the target domain/use case?
- Specificity: Are the principles actionable or vague?
- Confidence: Weight higher-confidence extractions

Respond with ONLY a JSON object:
{
  "selections": {
    "structural": {"source_id": "...", "score": 85, "reasoning": "..."},
    "linguistic": {"source_id": "...", "score": 90, "reasoning": "..."},
    ...for all 8 paradigms...
  },
  "conflicts": [
    {
      "paradigm_a": "structural",
      "paradigm_b": "rhythm",
      "conflict_description": "structural says brief recovery, rhythm says verbose",
      "resolution": "adapt rhythm to allow brief in recovery contexts",
      "priority_winner": "structural"
    }
  ]
}"""


async def mix_paradigms(state: PipelineState) -> dict:
    """LangGraph node: query DNA store, score, select, mix, and produce MixedDNA."""

    context_schema = state.get("context_schema", {})
    domain = context_schema.get("domain", "")

    if not domain:
        logger.warning("[ParadigmMixer] No domain in context, producing cold-start MixedDNA")
        return {
            "mixed_dna": MixedDNA(is_cold_start=True).model_dump(),
            "progress": "Paradigm mixing skipped (no domain)",
        }

    # Get all DNA records — first try domain-specific, then cross-domain
    dna_records = await sqlite_db.get_dna_for_domain(domain)
    cross_domain_records = await sqlite_db.get_all_dna_records(limit=30)

    # Merge and deduplicate
    seen_ids = {r.get("source_prompt_id") for r in dna_records}
    for r in cross_domain_records:
        if r.get("source_prompt_id") not in seen_ids:
            dna_records.append(r)
            seen_ids.add(r.get("source_prompt_id"))

    if not dna_records:
        logger.info("[ParadigmMixer] No DNA records available, cold start")
        return {
            "mixed_dna": MixedDNA(is_cold_start=True).model_dump(),
            "progress": "Paradigm mixing: cold start (no DNA in KB)",
        }

    # Build context for LLM selection
    persona = context_schema.get("persona", "")
    summary = context_schema.get("summary", "")
    tone = context_schema.get("tone", "")

    dna_summaries = []
    for i, rec in enumerate(dna_records[:15]):  # cap at 15 sources
        source_id = rec.get("source_prompt_id", f"unknown_{i}")
        rec_domain = rec.get("domain", "unknown")

        paradigm_summary = []
        for p in PARADIGM_NAMES:
            dna_field = rec.get(f"{p}_dna", {})
            principles = dna_field.get("principles", [])
            confidence = dna_field.get("confidence", 0.0)
            if principles:
                paradigm_summary.append(
                    f"    {p} (conf={confidence:.2f}): {'; '.join(principles[:3])}"
                )

        dna_summaries.append(
            f"SOURCE: {source_id} (domain: {rec_domain})\n"
            + "\n".join(paradigm_summary)
        )

    dna_block = "\n\n".join(dna_summaries)

    user_msg = f"""TARGET DOMAIN: {domain}
TARGET PERSONA: {persona}
TARGET TONE: {tone}
TARGET SUMMARY: {summary}

AVAILABLE DNA SOURCES ({len(dna_records)} records):

{dna_block}

Select the best DNA source for each of the 8 paradigms independently.
They can come from different sources. Detect and resolve conflicts."""

    llm = get_llm(max_tokens=2000)

    try:
        response = await llm.ainvoke([
            SystemMessage(content=MIXER_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        parsed = extract_json(response.content)
        if isinstance(parsed, list):
            parsed = {"selections": {}, "conflicts": []}
    except Exception as e:
        logger.warning(f"[ParadigmMixer] LLM call failed: {e}")
        # Fallback: use highest-confidence per paradigm
        return _fallback_mix(dna_records)

    selections = parsed.get("selections", {})
    conflicts_raw = parsed.get("conflicts", [])

    # Build MixedDNA from selections
    mixed = MixedDNA()
    source_map = {}

    # Index DNA records by source_prompt_id for lookup
    dna_by_id = {}
    for rec in dna_records:
        sid = rec.get("source_prompt_id", "")
        dna_by_id[sid] = rec

    for paradigm in PARADIGM_NAMES:
        sel = selections.get(paradigm, {})
        source_id = sel.get("source_id", "")

        if source_id and source_id in dna_by_id:
            source_rec = dna_by_id[source_id]
            dna_field = source_rec.get(f"{paradigm}_dna", {})
            pp = ParadigmPrinciples(
                paradigm=paradigm,
                principles=dna_field.get("principles", []),
                confidence=float(dna_field.get("confidence", 0.5)),
            )
            setattr(mixed, paradigm, pp)
            source_map[paradigm] = source_id
        else:
            # Fallback: use highest confidence for this paradigm
            best = _find_best_for_paradigm(dna_records, paradigm)
            if best:
                setattr(mixed, paradigm, best["pp"])
                source_map[paradigm] = best["source_id"]

    mixed.source_map = source_map

    # Parse conflicts
    for c in conflicts_raw:
        if isinstance(c, dict):
            try:
                mixed.conflicts.append(ParadigmConflict(**c))
            except Exception:
                pass

    total_principles = sum(len(mixed.get_paradigm(p).principles) for p in PARADIGM_NAMES)
    unique_sources = len(set(source_map.values()))

    logger.info(
        f"[ParadigmMixer] Mixed {total_principles} principles from "
        f"{unique_sources} sources, {len(mixed.conflicts)} conflicts resolved"
    )

    return {
        "mixed_dna": mixed.model_dump(),
        "progress": f"Paradigm mixing complete ({total_principles} principles from {unique_sources} sources)",
    }


def _find_best_for_paradigm(
    dna_records: list[dict], paradigm: str
) -> dict | None:
    """Find the DNA record with highest confidence for a given paradigm."""
    best = None
    best_confidence = -1.0

    for rec in dna_records:
        dna_field = rec.get(f"{paradigm}_dna", {})
        confidence = float(dna_field.get("confidence", 0.0))
        principles = dna_field.get("principles", [])

        if principles and confidence > best_confidence:
            best_confidence = confidence
            best = {
                "pp": ParadigmPrinciples(
                    paradigm=paradigm,
                    principles=principles,
                    confidence=confidence,
                ),
                "source_id": rec.get("source_prompt_id", ""),
            }

    return best


def _fallback_mix(dna_records: list[dict]) -> dict:
    """Fallback: use highest-confidence per paradigm without LLM scoring."""
    mixed = MixedDNA()
    source_map = {}

    for paradigm in PARADIGM_NAMES:
        best = _find_best_for_paradigm(dna_records, paradigm)
        if best:
            setattr(mixed, paradigm, best["pp"])
            source_map[paradigm] = best["source_id"]

    mixed.source_map = source_map

    total = sum(len(mixed.get_paradigm(p).principles) for p in PARADIGM_NAMES)
    logger.info(f"[ParadigmMixer] Fallback mix: {total} principles")

    return {
        "mixed_dna": mixed.model_dump(),
        "progress": f"Paradigm mixing (fallback): {total} principles",
    }
