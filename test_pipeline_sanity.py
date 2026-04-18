import asyncio
import json
from uuid import uuid4
from backend.graph.pipeline import build_pipeline_graph
from backend.models.schemas import PipelineState
from backend.kb import sqlite_db

async def main():
    await sqlite_db.init_db()
    
    run_id = str(uuid4())
    
    # Initialize run in DB to satisfy sqlite_db.update_run_progress calls
    import datetime
    created_at = datetime.datetime.now().isoformat()
    await sqlite_db.create_run(run_id, created_at)
    
    # Inject minimal state with multiple cases to test asyncio.gather concurrency
    state = PipelineState(
        run_id=run_id,
        context_doc="We are an IT helpdesk. Our persona is robotic and efficient. Just say hello and ask how we can help. If they have billing issues, tell them to check the portal. If technical, ask for error code.",
        state_names=["greeting", "billing_issue", "technical_support"],
        progress="starting"
    )
    
    from backend.graph.pipeline import get_compiled_graph
    app = get_compiled_graph()
    
    try:
        print("====== STARTING FAST HEADLESS PIPELINE RUN ======")
        async for event in app.astream(state):
            for node_name, node_state in event.items():
                print(f"Node '{node_name}' passed.")
                print(f"   -> Progress string: {node_state.get('progress')}")
                
        print("\n====== PIPELINE SUCCESS ======")
    except Exception as e:
        print(f"\n====== PIPELINE CRASHED ======")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
