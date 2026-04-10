"""Improver Agent 2 — State Decomposer improvement.
Triggers when: Missing whole case categories for the domain."""

from backend.agents.improver_base import execute_improvement as _base_improve

async def execute_improvement(feedback: str, is_upstream_cause: bool = False, upstream_agent: str = "") -> str:
    return await _base_improve("agent2", feedback, is_upstream_cause, upstream_agent)
