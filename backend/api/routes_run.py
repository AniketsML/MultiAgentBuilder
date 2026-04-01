"""API routes for pipeline execution: POST /run and GET /run/{id}/status."""

import asyncio
import json
import logging
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from backend.models.schemas import (
    RunRequest, RunStatusResponse, RunResult,
    ContextSchema, StateSpec, PromptDraft, PipelineState,
)
from backend.graph.state import make_initial_state
from backend.graph.pipeline import get_compiled_graph
from backend.kb import sqlite_db

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory store for active run results (supplement SQLite for fast access)
_active_runs: dict[str, dict] = {}


async def _execute_pipeline(run_id: str, initial_state: PipelineState):
    """Run the LangGraph pipeline in the background."""
    try:
        await sqlite_db.update_run_progress(run_id, "Starting pipeline...")

        graph = get_compiled_graph()

        # Stream through the graph nodes for progress updates
        final_state = None
        async for event in graph.astream(initial_state):
            # Each event is a dict with the node name as key
            for node_name, node_output in event.items():
                progress = node_output.get("progress", f"Running {node_name}...")
                await sqlite_db.update_run_progress(run_id, progress)
                # Merge into our tracking
                if run_id in _active_runs:
                    _active_runs[run_id]["progress"] = progress
            final_state = event

        # Build RunResult from the final state accumulated values
        if final_state:
            # Get the last node's output and merge with our tracking
            run_data = await sqlite_db.get_run(run_id)
            if run_data and run_data["status"] == "complete":
                _active_runs[run_id]["status"] = "complete"
            else:
                _active_runs[run_id]["status"] = "complete"

    except Exception as e:
        await sqlite_db.fail_run(run_id, str(e))
        if run_id in _active_runs:
            _active_runs[run_id]["status"] = "error"
            _active_runs[run_id]["error"] = str(e)


@router.post("/run")
async def start_run(request: RunRequest, background_tasks: BackgroundTasks):
    """Start a new pipeline run (async). Returns run_id immediately."""
    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    await sqlite_db.create_run(run_id, created_at)

    initial_state = make_initial_state(
        run_id=run_id,
        context_doc=request.context_doc,
        state_names=request.state_names,
        past_prompts=request.past_prompts,
    )

    _active_runs[run_id] = {
        "status": "running",
        "progress": "Starting...",
        "initial_state": initial_state,
        "error": None,
    }

    background_tasks.add_task(_execute_pipeline, run_id, initial_state)

    return {"run_id": run_id}


@router.get("/run/{run_id}/status")
async def get_run_status(run_id: str):
    """Poll for run progress. Returns current status and result when complete."""
    run_data = await sqlite_db.get_run(run_id)

    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")

    status = run_data["status"]
    progress = run_data.get("progress", "")
    error = run_data.get("error")
    result = None

    if status == "complete" and run_data.get("result_json"):
        result = json.loads(run_data["result_json"])

    return RunStatusResponse(
        status=status,
        progress=progress,
        result=RunResult(**result) if result else None,
        error=error,
    )


@router.get("/run/{run_id}/stream")
async def stream_run_progress(run_id: str):
    """SSE stream for live progress updates."""

    async def event_generator():
        prev_progress = ""
        while True:
            run_data = await sqlite_db.get_run(run_id)
            if not run_data:
                yield f"data: {json.dumps({'error': 'Run not found'})}\n\n"
                break

            current_progress = run_data.get("progress", "")
            status = run_data["status"]

            if current_progress != prev_progress:
                # If we are just yielding progress, don't send complete yet so the frontend doesn't close
                safe_status = "running" if status == "complete" else status
                yield f"data: {json.dumps({'status': safe_status, 'progress': current_progress})}\n\n"
                prev_progress = current_progress

            if status in ("complete", "error"):
                result = None
                
                print(f"[SSE Debug] Stream wrapping up. Status={status}, Has result_json={bool(run_data.get('result_json'))}")
                
                if status == "complete" and run_data.get("result_json"):
                    try:
                        result = json.loads(run_data["result_json"])
                        print(f"[SSE Debug] Successfully loaded result JSON with {len(result.get('drafts', []))} drafts.")
                    except Exception as e:
                        print(f"[SSE Debug] FAILED to parse result_json: {e}")
                elif status == "complete":
                    print("[SSE Debug] Status is complete but result_json is empty or missing in DB!")

                yield f"data: {json.dumps({'status': status, 'progress': current_progress, 'result': result, 'error': run_data.get('error')})}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
