"""LangGraph StateGraph pipeline -- case-based orchestrator with Prompt DNA.

New pipeline flow:
  analyse_context -> gate -> kb_case_learner -> gate ->
  paradigm_mixer -> gate -> decompose_states -> gate ->
  prioritise_cases -> gate -> seed_kb -> write_case_handlers -> gate ->
  assemble_prompts -> gate -> review_consistency -> finalise -> END

Review gates use per-gate specialized rubrics.
Cross-agent causation detection routes improvements to root cause.
Paradigm Mixer injects MixedDNA for downstream DNA compliance.
"""

import json
import logging
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END

from backend.models.schemas import PipelineState, ContextSchema, StateDecomposition
from backend.agents.agent1_analyser import analyse_context
from backend.agents.agent_kb_learner import learn_cases
from backend.agents.agent_paradigm_mixer import mix_paradigms
from backend.agents.agent2_planner import decompose_states
from backend.agents.agent_case_prioritiser import prioritise_cases
from backend.agents.agent_case_writer import write_case_handlers
from backend.agents.agent_assembler import assemble_prompts, reassemble_prompt
from backend.agents.agent4_reviewer import review_consistency
from backend.agents.review_agent import review_stage_output
from backend.agents.agent_master import run_master_chat
from backend.kb.kb_writer import seed_kb_if_new
from backend.kb import sqlite_db

logger = logging.getLogger(__name__)

# Maps stage names to their responsible agent ID for targeted improvements
STAGE_TO_AGENT = {
    "context_analysis": "agent1",
    "kb_case_learning": "kb_learner",
    "paradigm_mixing": "paradigm_mixer",
    "state_decomposition": "agent2",
    "case_prioritisation": "case_prioritiser",
    "case_writing": "case_writer",
    "prompt_assembly": "assembler",
    "consistency_review": "agent4",
}

# Maps agent IDs to their improver module names
AGENT_TO_IMPROVER = {
    "agent0": "improver_agent0",
    "agent1": "improver_agent1",
    "agent2": "improver_agent2",
    "kb_learner": "improver_kb_learner",
    "paradigm_mixer": "improver_paradigm_mixer",
    "case_prioritiser": "improver_case_prioritiser",
    "case_writer": "improver_case_writer",
    "assembler": "improver_assembler",
    "agent4": "improver_agent4",
    "dna_analyzer": "improver_dna_analyzer",
}

COLD_START_MAX_ITERATIONS = 2


# ──────────────── Worker Node Wrappers ────────────────

async def node_analyse(state: PipelineState) -> dict:
    """Wrapper for Agent 1."""
    return await analyse_context(state)


async def node_kb_case_learner(state: PipelineState) -> dict:
    """Wrapper for KB Case Learner."""
    return await learn_cases(state)


async def node_paradigm_mixer(state: PipelineState) -> dict:
    """Wrapper for Paradigm Mixer."""
    return await mix_paradigms(state)


async def node_decompose_states(state: PipelineState) -> dict:
    """Wrapper for Agent 2 (State Decomposer)."""
    return await decompose_states(state)


async def node_prioritise_cases(state: PipelineState) -> dict:
    """Wrapper for Case Prioritiser."""
    return await prioritise_cases(state)


async def node_seed_kb(state: PipelineState) -> dict:
    """Seed KB with past prompts if provided and domain not yet seeded."""
    past = state.get("past_prompts")
    if past:
        context = ContextSchema(**state["context_schema"])
        await seed_kb_if_new(past, context, state["run_id"])
    return {"progress": "KB seeded"}


async def node_write_case_handlers(state: PipelineState) -> dict:
    """Wrapper for Case Writer."""
    return await write_case_handlers(state)


async def node_assemble_prompts(state: PipelineState) -> dict:
    """Wrapper for Prompt Assembler."""
    return await assemble_prompts(state)


async def node_review(state: PipelineState) -> dict:
    """Wrapper for Agent 4."""
    return await review_consistency(state)


async def node_regenerate(state: PipelineState) -> dict:
    """Wrapper for regeneration (re-assembles from case handlers)."""
    return await reassemble_prompt(state)


# ──────────────── Review Gate Nodes ────────────────

async def _review_gate(
    state: PipelineState,
    stage_name: str,
    output_key: str,
) -> dict:
    """Generic review gate with per-gate specialized rubrics and cross-agent causation."""
    context_doc = state.get("context_doc", "") or state.get("raw_text", "")
    stage_output = state.get(output_key)
    is_cold_start = state.get("is_cold_start", False)

    if not stage_output or not context_doc:
        return {"progress": f"Review gate skipped for {stage_name} (no output)"}

    try:
        # Collect upstream scorecards for causation detection
        existing_scorecards = list(state.get("critic_scorecards", []))

        # For assembly gate, pass prioritised cases for cross-reference
        prioritised_cases = None
        if stage_name == "prompt_assembly":
            prioritised_cases = state.get("prioritised_cases", [])

        scorecard = await review_stage_output(
            stage_name,
            stage_output,
            context_doc,
            is_cold_start=is_cold_start,
            upstream_scorecards=existing_scorecards,
            prioritised_cases=prioritised_cases,
        )

        total = scorecard.get("total", 100)
        passed = scorecard.get("passed", True)
        failed_dims = scorecard.get("failed_dimensions", [])
        root_cause = scorecard.get("root_cause_agent", "")

        logger.info(
            f"[ReviewGate] {stage_name}: {total}/100 | passed={passed} | "
            f"failed_dims={failed_dims}"
            f"{' | root_cause=' + root_cause if root_cause else ''}"
        )

        existing_scorecards.append(scorecard)

        updates = {
            "critic_scorecards": existing_scorecards,
            "progress": f"Reviewed {stage_name}: {total}/100",
        }

        if not passed:
            targeted = scorecard.get("targeted_instructions", {})
            agent_id = STAGE_TO_AGENT.get(stage_name)

            # Determine primary improvement target
            primary_target = root_cause if root_cause else agent_id
            is_upstream = bool(root_cause and root_cause != agent_id)

            if targeted and primary_target:
                improver_name = AGENT_TO_IMPROVER.get(primary_target)
                if improver_name:
                    try:
                        improver_module = __import__(
                            f"backend.agents.{improver_name}",
                            fromlist=["execute_improvement"],
                        )

                        feedback_parts = []
                        for dim, instruction in targeted.items():
                            score = scorecard.get("dimensions", {}).get(dim, "?")
                            feedback_parts.append(
                                f"[{stage_name}] {dim} scored {score}/100. Fix: {instruction}"
                            )
                        feedback = "AUTO-REVIEW: " + " | ".join(feedback_parts)

                        logger.info(
                            f"[ReviewGate] Routing to {improver_name} "
                            f"(upstream={is_upstream}): {feedback[:120]}..."
                        )
                        await improver_module.execute_improvement(
                            feedback,
                            is_upstream_cause=is_upstream,
                            upstream_agent=root_cause if is_upstream else "",
                        )
                    except Exception as e:
                        logger.warning(
                            f"[ReviewGate] Improver {improver_name} failed (non-fatal): {e}"
                        )

                # Also route to the scored agent if root cause is upstream
                if is_upstream and agent_id and agent_id != primary_target:
                    secondary_improver = AGENT_TO_IMPROVER.get(agent_id)
                    if secondary_improver:
                        try:
                            sec_module = __import__(
                                f"backend.agents.{secondary_improver}",
                                fromlist=["execute_improvement"],
                            )
                            sec_feedback = (
                                f"AUTO-REVIEW (secondary): {stage_name} failed. "
                                f"Root cause is upstream ({root_cause}), but add resilience."
                            )
                            await sec_module.execute_improvement(
                                sec_feedback, is_upstream_cause=True, upstream_agent=root_cause
                            )
                        except Exception:
                            pass

            # Notify Master Agent
            summary = (
                f"AUTO-REVIEW: {stage_name} scored {total}/100. "
                f"Failed: {', '.join(failed_dims)}"
                f"{f'. Root cause: {root_cause}' if root_cause else ''}"
            )
            try:
                await run_master_chat(summary)
            except Exception:
                pass

        return updates

    except Exception as e:
        logger.warning(f"[ReviewGate] {stage_name} review failed (non-fatal): {e}")
        return {"progress": f"Review gate for {stage_name} skipped (error)"}


async def gate_after_analyse(state: PipelineState) -> dict:
    return await _review_gate(state, "context_analysis", "context_schema")


async def gate_after_kb_learning(state: PipelineState) -> dict:
    return await _review_gate(state, "kb_case_learning", "case_learning_contexts")


async def gate_after_decompose(state: PipelineState) -> dict:
    return await _review_gate(state, "state_decomposition", "state_decompositions")


async def gate_after_prioritise(state: PipelineState) -> dict:
    return await _review_gate(state, "case_prioritisation", "prioritised_cases")


async def gate_after_case_write(state: PipelineState) -> dict:
    return await _review_gate(state, "case_writing", "case_handlers")


async def gate_after_assembly(state: PipelineState) -> dict:
    return await _review_gate(state, "prompt_assembly", "drafts")


async def gate_after_paradigm_mixer(state: PipelineState) -> dict:
    return await _review_gate(state, "paradigm_mixing", "mixed_dna")


# ──────────────── Finalise ────────────────

async def node_finalise(state: PipelineState) -> dict:
    """Mark the pipeline as complete and store results in SQLite."""
    from backend.models.schemas import RunResult, ContextSchema, StateDecomposition, PromptDraft, VariableSchema

    context = ContextSchema(**state["context_schema"])
    decompositions = [StateDecomposition(**d) for d in state.get("state_decompositions", [])]
    drafts = [PromptDraft(**d) for d in state.get("drafts", [])]

    # Aggregate variables from all decompositions
    all_vars = []
    seen_names = set()
    for d in decompositions:
        for v in d.extracted_variables:
            if v.name not in seen_names:
                seen_names.add(v.name)
                all_vars.append(v)

    result = RunResult(
        run_id=state["run_id"],
        context=context,
        states=decompositions,
        variables=all_vars,
        drafts=drafts,
        review_notes=state.get("review_notes", ""),
        case_learning_contexts=state.get("case_learning_contexts", []),
        prioritised_cases=state.get("prioritised_cases", []),
        case_handlers=state.get("case_handlers", []),
        mixed_dna=state.get("mixed_dna"),
    )

    await sqlite_db.complete_run(state["run_id"], result.model_dump_json())

    return {"progress": "Pipeline complete"}


# ──────────────── Build the graph ────────────────

def build_pipeline_graph() -> StateGraph:
    """Construct the case-based pipeline with specialized review gates."""

    graph = StateGraph(PipelineState)

    # Worker nodes
    graph.add_node("analyse_context", node_analyse)
    graph.add_node("kb_case_learner", node_kb_case_learner)
    graph.add_node("paradigm_mixer", node_paradigm_mixer)
    graph.add_node("decompose_states", node_decompose_states)
    graph.add_node("prioritise_cases", node_prioritise_cases)
    graph.add_node("seed_kb", node_seed_kb)
    graph.add_node("write_case_handlers", node_write_case_handlers)
    graph.add_node("assemble_prompts", node_assemble_prompts)
    graph.add_node("review_consistency", node_review)
    graph.add_node("finalise", node_finalise)

    # Review gate nodes
    graph.add_node("gate_after_analyse", gate_after_analyse)
    graph.add_node("gate_after_kb_learning", gate_after_kb_learning)
    graph.add_node("gate_after_paradigm_mixer", gate_after_paradigm_mixer)
    graph.add_node("gate_after_decompose", gate_after_decompose)
    graph.add_node("gate_after_prioritise", gate_after_prioritise)
    graph.add_node("gate_after_case_write", gate_after_case_write)
    graph.add_node("gate_after_assembly", gate_after_assembly)

    # Flow:
    # analyse -> gate -> kb_case_learner -> gate -> paradigm_mixer -> gate ->
    # decompose -> gate -> prioritise -> gate -> seed_kb -> write_handlers -> gate ->
    # assemble -> gate -> review_consistency -> finalise -> END
    graph.set_entry_point("analyse_context")
    graph.add_edge("analyse_context", "gate_after_analyse")
    graph.add_edge("gate_after_analyse", "kb_case_learner")
    graph.add_edge("kb_case_learner", "gate_after_kb_learning")
    graph.add_edge("gate_after_kb_learning", "paradigm_mixer")
    graph.add_edge("paradigm_mixer", "gate_after_paradigm_mixer")
    graph.add_edge("gate_after_paradigm_mixer", "decompose_states")
    graph.add_edge("decompose_states", "gate_after_decompose")
    graph.add_edge("gate_after_decompose", "prioritise_cases")
    graph.add_edge("prioritise_cases", "gate_after_prioritise")
    graph.add_edge("gate_after_prioritise", "seed_kb")
    graph.add_edge("seed_kb", "write_case_handlers")
    graph.add_edge("write_case_handlers", "gate_after_case_write")
    graph.add_edge("gate_after_case_write", "assemble_prompts")
    graph.add_edge("assemble_prompts", "gate_after_assembly")
    graph.add_edge("gate_after_assembly", "review_consistency")
    graph.add_edge("review_consistency", "finalise")
    graph.add_edge("finalise", END)

    return graph


def get_compiled_graph():
    """Return a compiled graph ready for invocation."""
    graph = build_pipeline_graph()
    return graph.compile()


# ──────────────── Regeneration graph ────────────────

def build_regen_graph() -> StateGraph:
    """Build a sub-graph for regeneration.

    Re-runs from Prompt Assembler with user feedback incorporated.
    """
    graph = StateGraph(PipelineState)
    graph.add_node("regenerate", node_regenerate)
    graph.set_entry_point("regenerate")
    graph.add_edge("regenerate", END)
    return graph


def get_compiled_regen_graph():
    """Return a compiled regen graph."""
    graph = build_regen_graph()
    return graph.compile()
