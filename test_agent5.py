import asyncio
import sys
import os

# Add the project root to sys.path so backend module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.agents.agent5_extractor import extract_variables
from backend.models.schemas import PipelineState

async def test_agent5():
    state = PipelineState(
        run_id="test_123",
        context_doc="We need a voice bot for debt collection. The bot should call the customer and ask for their first name, last name, and the amount they want to pay today. If they agree to pay, we should also collect their payment date.",
        state_names=[],
        context_schema=None,
        state_specs=[],
        current_state_index=0,
        drafts=[],
        review_findings=[],
        progress="",
        error=None,
        regen_state_name=None,
        regen_reason=None
    )
    result = await extract_variables(state)
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test_agent5())
