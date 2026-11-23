from uuid import uuid7

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic_core import to_jsonable_python
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.state import State

from webkiln_api.models import RunDetail, RunRecord, RunResponse, thread_config
from webkiln_api.streaming import finished_events, run_events

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", status_code=201)
async def create_run(brief: Input, request: Request) -> RunResponse:
    thread_id = str(uuid7())
    record = RunRecord(brief=brief)
    request.app.state.runs[thread_id] = record
    return RunResponse(thread_id=thread_id, status=record.status)


@router.get("/{thread_id}")
async def get_run(thread_id: str, request: Request) -> RunDetail:
    record = request.app.state.runs.get(thread_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    snapshot = await request.app.state.graph.aget_state(thread_config(thread_id))
    # Only the fields the graph state declares. Pydantic objects become plain JSON.
    values = {k: v for k, v in snapshot.values.items() if k in State.model_fields}
    return RunDetail(
        thread_id=thread_id,
        status=record.status,
        brief=record.brief,
        state=to_jsonable_python(values) if values else None,
    )


@router.get("/{thread_id}/stream")
async def stream_run(thread_id: str, request: Request) -> StreamingResponse:
    """Start the run and stream its progress as server-sent events."""
    record = request.app.state.runs.get(thread_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if record.status == "running":
        raise HTTPException(status_code=409, detail="Run is already streaming")

    if record.status == "created":
        record.status = "running"  # set here, so a second connection is refused at once
        events = run_events(request.app.state.graph, record, thread_id)
    else:
        events = finished_events(record)
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(events, media_type="text/event-stream", headers=headers)
