"""Improver for the Paradigm Mixer agent."""
from backend.agents.improver_base import execute_improvement as _base_improve

async def execute_improvement(feedback: str, **kwargs):
    await _base_improve("paradigm_mixer", feedback, **kwargs)
