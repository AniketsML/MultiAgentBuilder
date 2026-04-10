"""Improver for Case Prioritiser.
Triggers when: Filtering cases that turned out to be important."""

from backend.agents.improver_base import execute_improvement as _base_improve

async def execute_improvement(feedback: str, is_upstream_cause: bool = False, upstream_agent: str = "") -> str:
    return await _base_improve("case_prioritiser", feedback, is_upstream_cause, upstream_agent)
