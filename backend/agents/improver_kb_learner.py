"""Improver for KB Case Learner.
Triggers when: Not extracting the right case categories from KB."""

from backend.agents.improver_base import execute_improvement as _base_improve

async def execute_improvement(feedback: str, is_upstream_cause: bool = False, upstream_agent: str = "") -> str:
    return await _base_improve("kb_learner", feedback, is_upstream_cause, upstream_agent)
