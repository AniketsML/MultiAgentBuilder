"""Improver for the DNA Analyzer agent."""
from backend.agents.improver_base import execute_improvement as _base_improve

async def execute_improvement(feedback: str, **kwargs):
    await _base_improve("dna_analyzer", feedback, **kwargs)
