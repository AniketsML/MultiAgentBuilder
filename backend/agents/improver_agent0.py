"""Improver Agent 0 — State Extractor improvement.
Triggers when: States are too granular or too coarse."""

from backend.agents.improver_base import execute_improvement as _base_improve

async def execute_improvement(feedback: str, is_upstream_cause: bool = False, upstream_agent: str = "") -> str:
    return await _base_improve("agent0", feedback, is_upstream_cause, upstream_agent)
