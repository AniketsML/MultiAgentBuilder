"""Improver for Prompt Assembler.
Triggers when: Final prompt is incoherent or over limit."""

from backend.agents.improver_base import execute_improvement as _base_improve

async def execute_improvement(feedback: str, is_upstream_cause: bool = False, upstream_agent: str = "") -> str:
    return await _base_improve("assembler", feedback, is_upstream_cause, upstream_agent)
