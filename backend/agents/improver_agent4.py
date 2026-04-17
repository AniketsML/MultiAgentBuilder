"""Improver Agent 4 — Consistency Reviewer improvement.
Triggers when: Missing cross-state inconsistencies."""

from backend.agents.improver_base import execute_improvement as _base_improve

async def execute_improvement(feedback: str, is_upstream_cause: bool = False, upstream_agent: str = "") -> str:
    return await _base_improve("agent4", feedback, is_upstream_cause, upstream_agent)
