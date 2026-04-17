"""Improver for the Paradigm Mixer agent."""
from backend.agents.improver_base import execute_base_improvement

async def execute_improvement(feedback: str, **kwargs):
    await execute_base_improvement("paradigm_mixer", feedback, **kwargs)
