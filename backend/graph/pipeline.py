"""LangGraph StateGraph pipeline — main orchestrator.

Builds the graph with nodes for all worker agents, per-stage review gates
with per-dimension critic scoring, the Pattern Abstractor layer, and
cold-start-aware behavior. Review gates route targeted_instructions
directly to Improver Agents, bypassing Master Agent for targeted fixes.
"""

import json
import logging
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END

from backend.models.schemas import PipelineState, ContextSchema
from backend.agents.agent1_analyser import analyse_context
from backend.agents.agent2_planner import plan_states
from backend.agents.agent3_writer import write_prompts, regenerate_prompt
from backend.agents.agent4_reviewer import review_consistency
from backend.agents.agent5_extractor import extract_variables
from backend.agents.agent6_pattern_abstractor import abstract_patterns
from backend.agents.review_agent import review_stage_output
from backend.agents.agent_master import run_master_chat
from backend.kb.kb_writer import seed_kb_if_new
from backend.kb import sqlite_db

logger = logging.getLogger(__name__)

# Maps stage names to their responsible agent ID for targeted improvements
STAGE_TO_AGENT = {
    "context_analysis": "agent1",
    "pattern_abstraction": "agent6",
    "state_planning": "agent2",
    "variable_extraction": "agent5",
    "prompt_writing": "agent3",
}

# Cold start: max improvement iterations capped at 2 (not 3) to control cost
COLD_START_MAX_ITERATIONS = 2


# ──────────────────── Worker Node Wrappers ────────────────────

async def node_analyse(state: PipelineState) -> dict:
    """Wrapper for Agent 1."""
    return await analyse_context(state)


async def node_pattern_abstractor(state: PipelineState) -> dict:
    """Wrapper for Agent 6 — Pattern Abstractor."""
    return await abstract_patterns(state)


async def node_extract_variables(state: PipelineState) -> dict:
    """Wrapper for Agent 5."""
    return await extract_variables(state)


async def node_plan(state: PipelineState) -> dict:
    """Wrapper for Agent 2."""
    return await plan_states(state)


async def node_seed_kb(state: PipelineState) -> dict:
    """Seed KB with past prompts if provided and domain not yet seeded."""
    past = state.get("past_prompts")
    if past:
        context = ContextSchema(**state["context_schema"])
        await seed_kb_if_new(past, context, state["run_id"])
    return {"progress": "KB seeded"}


async def node_write_prompts(state: PipelineState) -> dict:
    """Wrapper for Agent 3 — writes all prompts."""
    return await write_prompts(state)


async def node_review(state: PipelineState) -> dict:
    """Wrapper for Agent 4."""
    return await review_consistency(state)


async def node_regenerate(state: PipelineState) -> dict:
    """Wrapper for Agent 3 regeneration."""
    return await regenerate_prompt(state)


# ──────────────────── Review Gate Nodes ────────────────────
# Each gate: reviews the previous stage output → per-dimension scoring →
# routes targeted_instructions directly to Improver Agents

async def _review_gate(state: PipelineState, stage_name: str, output_key: str) -> dict:
    """Generic review gate with per-dimension critic scoring."""
    context_doc = state.get("context_doc", "") or state.get("raw_text", "")
    stage_output = state.get(output_key)
    is_cold_start = state.get("is_cold_start", False)

    if not stage_output or not context_doc:
        return {"progress": f"Review gate skipped for {stage_name} (no output)"}

    try:
        scorecard = await review_stage_output(
            stage_name, stage_output, context_doc, is_cold_start=is_cold_start
        )

        total = scorecard.get("total", 100)
        passed = scorecard.get("passed", True)
        failed_dims = scorecard.get("failed_dimensions", [])

        logger.info(
            f"[ReviewGate] {stage_name}: {total}/100 | passed={passed} | "
            f"failed_dims={failed_dims}"
        )

        # Store scorecard
        existing_scorecards = list(state.get("critic_scorecards", []))
        existing_scorecards.append(scorecard)

        updates = {
            "critic_scorecards": existing_scorecards,
            "progress": f"Reviewed {stage_name}: {total}/100",
        }

        if not passed:
            targeted = scorecard.get("targeted_instructions", {})
            agent_id = STAGE_TO_AGENT.get(stage_name)

            if targeted and agent_id:
                # Route targeted instructions directly to the responsible Improver
                for dim, instruction in targeted.items():
                    try:
                        improver_module = __import__(
                            f"backend.agents.improver_{agent_id}",
                            fromlist=["execute_improvement"],
                        )
                        feedback = (
                            f"AUTO-REVIEW [{stage_name}]: {dim} scored "
                            f"{scorecard['dimensions'].get(dim, '?')}/100. "
                            f"Fix: {instruction}"
                        )
                        logger.info(
                            f"[ReviewGate] Routing to improver_{agent_id}: {feedback[:120]}..."
                        )
                        await improver_module.execute_improvement(feedback)
                    except Exception as e:
                        logger.warning(
                            f"[ReviewGate] Improver {agent_id} failed (non-fatal): {e}"
                        )

            # Notify Master Agent with summary only (not repair instructions)
            summary = (
                f"AUTO-REVIEW: {stage_name} scored {total}/100. "
                f"Failed dimensions: {', '.join(failed_dims)}"
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

async def gate_after_pattern(state: PipelineState) -> dict:
    """Review gate for Pattern Abstractor with validation + retry."""
    result = await _review_gate(state, "pattern_abstraction", "pattern_analysis")

    # Validate Pattern Abstractor output — retry once if invalid
    pattern = state.get("pattern_analysis", {})
    if pattern:
        skeleton = pattern.get("template_skeleton", "")
        rules = pattern.get("core_rules", [])

        if not skeleton or len(rules) < 2:
            logger.warning(
                "[ReviewGate] Pattern Abstractor output invalid "
                f"(skeleton={bool(skeleton)}, rules={len(rules)}). Retrying once..."
            )
            try:
                # Fire improver_agent6 to fix the prompt
                from backend.agents import improver_agent6
                await improver_agent6.execute_improvement(
                    "Output was invalid: template_skeleton was empty or core_rules had fewer than 2 entries. "
                    "Make the prompt more explicit about requiring these fields."
                )
                # Re-run Pattern Abstractor
                from backend.agents.agent6_pattern_abstractor import abstract_patterns
                retry_result = await abstract_patterns(state)
                result.update(retry_result)
            except Exception as e:
                logger.warning(f"[ReviewGate] Pattern Abstractor retry failed: {e}")

    return result

async def gate_after_plan(state: PipelineState) -> dict:
    return await _review_gate(state, "state_planning", "state_specs")

async def gate_after_variables(state: PipelineState) -> dict:
    return await _review_gate(state, "variable_extraction", "extracted_variables")

async def gate_after_write(state: PipelineState) -> dict:
    return await _review_gate(state, "prompt_writing", "drafts")


# ──────────────────── Finalise ────────────────────

async def node_finalise(state: PipelineState) -> dict:
    """Mark the pipeline as complete and store results in SQLite."""
    from backend.models.schemas import RunResult, ContextSchema, StateSpec, PromptDraft

    context = ContextSchema(**state["context_schema"])
    specs = [StateSpec(**s) for s in state["state_specs"]]
    drafts = [PromptDraft(**d) for d in state["drafts"]]

    result = RunResult(
        run_id=state["run_id"],
        context=context,
        states=specs,
        variables=state.get("extracted_variables", []),
        drafts=drafts,
        review_notes=state.get("review_notes", ""),
    )

    await sqlite_db.complete_run(state["run_id"], result.model_dump_json())

    return {"progress": "Pipeline complete"}


# ──────────────────── Build the graph ────────────────────

def build_pipeline_graph() -> StateGraph:
    """Construct the pipeline with Pattern Abstractor and review gates."""

    graph = StateGraph(PipelineState)

    # Worker nodes
    graph.add_node("analyse_context", node_analyse)
    graph.add_node("pattern_abstractor", node_pattern_abstractor)
    graph.add_node("plan_states", node_plan)
    graph.add_node("extract_variables", node_extract_variables)
    graph.add_node("seed_kb", node_seed_kb)
    graph.add_node("write_prompts", node_write_prompts)
    graph.add_node("review_consistency", node_review)
    graph.add_node("finalise", node_finalise)

    # Review gate nodes
    graph.add_node("gate_after_analyse", gate_after_analyse)
    graph.add_node("gate_after_pattern", gate_after_pattern)
    graph.add_node("gate_after_plan", gate_after_plan)
    graph.add_node("gate_after_variables", gate_after_variables)
    graph.add_node("gate_after_write", gate_after_write)

    # Flow: analyse → gate → pattern_abstractor → gate → plan → gate →
    #        extract_vars → gate → seed_kb → write → gate → review → finalise
    graph.set_entry_point("analyse_context")
    graph.add_edge("analyse_context", "gate_after_analyse")
    graph.add_edge("gate_after_analyse", "pattern_abstractor")
    graph.add_edge("pattern_abstractor", "gate_after_pattern")
    graph.add_edge("gate_after_pattern", "plan_states")
    graph.add_edge("plan_states", "gate_after_plan")
    graph.add_edge("gate_after_plan", "extract_variables")
    graph.add_edge("extract_variables", "gate_after_variables")
    graph.add_edge("gate_after_variables", "seed_kb")
    graph.add_edge("seed_kb", "write_prompts")
    graph.add_edge("write_prompts", "gate_after_write")
    graph.add_edge("gate_after_write", "review_consistency")
    graph.add_edge("review_consistency", "finalise")
    graph.add_edge("finalise", END)

    return graph


def get_compiled_graph():
    """Return a compiled graph ready for invocation."""
    graph = build_pipeline_graph()
    return graph.compile()


# ──────────────────── Regeneration graph ────────────────────

def build_regen_graph() -> StateGraph:
    """Build a small sub-graph just for prompt regeneration."""
    graph = StateGraph(PipelineState)
    graph.add_node("regenerate", node_regenerate)
    graph.set_entry_point("regenerate")
    graph.add_edge("regenerate", END)
    return graph


def get_compiled_regen_graph():
    """Return a compiled regen graph."""
    graph = build_regen_graph()
    return graph.compile()
