from typing import Annotated
from uuid import uuid7

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langgraph.types import StateSnapshot
from pydantic_core import to_jsonable_python
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.state import State

from webkiln_api.models import RunDetail, RunRecord, RunResponse, RunSummary, thread_config
from webkiln_api.openapi import ERROR_RESPONSES
from webkiln_api.streaming import finished_events, run_events

router = APIRouter(prefix="/runs", tags=["runs"], responses=ERROR_RESPONSES)


def record_from_snapshot(snapshot: StateSnapshot) -> RunRecord | None:
    """Rebuild a run record from a saved checkpoint, or None when the thread has no run."""
    brief = snapshot.values.get("initial_input")
    if brief is None:
        return None
    # Nothing left to run means it finished. A pending step means the old process died mid-run.
    return RunRecord(brief=brief, status="failed" if snapshot.next else "completed")


async def find_record(request: Request, thread_id: str) -> RunRecord | None:
    """Return the run record, rebuilding it from the saved checkpoint after an API restart.

    A run that was created but never started has no checkpoint, so a restart forgets it.
    """
    runs: dict[str, RunRecord] = request.app.state.runs
    if thread_id not in runs:
        snapshot = await request.app.state.graph.aget_state(thread_config(thread_id))
        record = record_from_snapshot(snapshot)
        if record is None:
            return None
        runs[thread_id] = record
    return runs[thread_id]


@router.post("", status_code=201)
async def create_run(brief: Input, request: Request) -> RunResponse:
    thread_id = str(uuid7())
    record = RunRecord(brief=brief)
    request.app.state.runs[thread_id] = record
    return RunResponse(thread_id=thread_id, status=record.status)


@router.get("")
async def list_runs(
    request: Request, limit: Annotated[int, Query(ge=1, le=50)] = 20
) -> list[RunSummary]:
    """The saved runs, newest first."""
    graph = request.app.state.graph
    # The saver lists checkpoints newest first, so a thread's first hit is its latest.
    latest: dict[str, str] = {}
    async for saved in graph.checkpointer.alist(None):
        thread_id = saved.config["configurable"]["thread_id"]
        if thread_id not in latest:
            latest[thread_id] = saved.checkpoint["ts"]
            if len(latest) == limit:
                break

    summaries = list[RunSummary]()
    for thread_id, updated_at in latest.items():
        record = await find_record(request, thread_id)
        if record is not None:
            summaries.append(
                RunSummary(
                    thread_id=thread_id,
                    status=record.status,
                    brief=record.brief,
                    updated_at=updated_at,
                )
            )
    return summaries


@router.get("/{thread_id}")
async def get_run(thread_id: str, request: Request) -> RunDetail:
    record = await find_record(request, thread_id)
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
    record = await find_record(request, thread_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if record.status == "running":
        raise HTTPException(status_code=409, detail="Run is already streaming")

    if record.status == "created":
        record.status = "running"  # set here, so a second connection is refused at once
        events = run_events(request.app.state.graph, record, thread_id, request.app.state.sessions)
    else:
        events = finished_events(record)
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(events, media_type="text/event-stream", headers=headers)
