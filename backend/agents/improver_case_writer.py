"""Improver Agent 3 — Case Writer improvement (was Prompt Writer).
Triggers when: Case handlers are weak or inconsistent."""

from backend.agents.improver_base import execute_improvement as _base_improve

async def execute_improvement(feedback: str, is_upstream_cause: bool = False, upstream_agent: str = "") -> str:
    return await _base_improve("case_writer", feedback, is_upstream_cause, upstream_agent)
