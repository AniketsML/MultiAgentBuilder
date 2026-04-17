"""Improver Agent 1 — Context Analyser improvement.
Triggers when: Guardrails or persona misses domain nuance."""

from backend.agents.improver_base import execute_improvement as _base_improve

async def execute_improvement(feedback: str, is_upstream_cause: bool = False, upstream_agent: str = "") -> str:
    return await _base_improve("agent1", feedback, is_upstream_cause, upstream_agent)
