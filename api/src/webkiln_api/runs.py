from typing import Annotated, cast
from uuid import uuid7

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic_core import to_jsonable_python
from webkiln_agents.budget import CALL_ALLOWANCE
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.plan import Task
from webkiln_agents.manager.review import ReviewDecision
from webkiln_agents.manager.state import State

from webkiln_api import store
from webkiln_api.models import RunDetail, RunResponse, RunSummary, TaskReport, thread_config
from webkiln_api.openapi import ERROR_RESPONSES
from webkiln_api.site import without_edit_files, without_files
from webkiln_api.status import STUCK
from webkiln_api.store import StoredRun
from webkiln_api.streaming import finished_events
from webkiln_api.supervisor import RunSupervisor

# Becomes a namespace label in the memory store, which cannot hold a period.
USER_ID_PATTERN = r"^[A-Za-z0-9_-]{1,100}$"

router = APIRouter(prefix="/runs", tags=["runs"], responses=ERROR_RESPONSES)


def supervisor_of(request: Request) -> RunSupervisor:
    return request.app.state.supervisor


async def get_stored_run(request: Request, thread_id: str) -> StoredRun:
    run = await store.find_run(request.app.state.sessions, thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("", status_code=201)
async def create_run(
    brief: Input,
    request: Request,
    user_id: Annotated[str | None, Header(alias="X-User-Id", pattern=USER_ID_PATTERN)] = None,
) -> RunResponse:
    """Save the run and start it in the background. Follow it with the stream route.

    `X-User-Id` says whose long-term memory the run reads and writes (letters, digits, `_` and
    `-`, up to 100). Without it the run has no memory.
    """
    thread_id = str(uuid7())
    await store.create_run_rows(request.app.state.sessions, thread_id, brief, user_id)
    supervisor_of(request).start(thread_id, brief)
    return RunResponse(thread_id=thread_id, status="running")


@router.get("")
async def list_runs(
    request: Request, limit: Annotated[int, Query(ge=1, le=50)] = 20
) -> list[RunSummary]:
    """The saved runs, changed most recently first."""
    runs = await store.list_runs(request.app.state.sessions, limit)
    return [
        RunSummary(
            thread_id=run.thread_id,
            status=run.status,
            brief=run.brief,
            updated_at=run.updated_at.isoformat(),
        )
        for run in runs
    ]


@router.get("/{thread_id}")
async def get_run(thread_id: str, request: Request) -> RunDetail:
    run = await get_stored_run(request, thread_id)
    snapshot = await request.app.state.graph.aget_state(thread_config(thread_id))
    # Only the fields the graph state declares. Pydantic objects become plain JSON.
    values = {k: v for k, v in snapshot.values.items() if k in State.model_fields}
    if "outputs" in values:  # the files of a site are served by the site routes, not the detail
        values["outputs"] = {task: without_files(out) for task, out in values["outputs"].items()}
    values = without_edit_files(values)
    # What the user is asked to decide. It comes from the checkpoint, so a restart keeps it.
    review = snapshot.interrupts[0].value if snapshot.interrupts else None
    return RunDetail(
        thread_id=thread_id,
        status=run.status,
        brief=run.brief,
        state=to_jsonable_python(values) if values else None,
        review=to_jsonable_python(review) if review else None,
        model_calls=run.model_calls,
        call_limit=run.call_limit,
    )


@router.get("/{thread_id}/stream")
async def stream_run(thread_id: str, request: Request) -> StreamingResponse:
    """Follow a run as server-sent events.

    A run that is still going sends its events so far and then the live ones, so this can be
    called again after a page refresh. A run that is not going only sends `done`.
    """
    run = await get_stored_run(request, thread_id)
    channel = supervisor_of(request).channels.get(thread_id)
    events = channel.subscribe() if channel else finished_events(run.status)
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(events, media_type="text/event-stream", headers=headers)


async def restart_stuck_tasks(request: Request, thread_id: str) -> RunResponse:
    """Put the failed tasks and the ones that wait for the user back to `pending`, and run on."""
    supervisor = supervisor_of(request)
    graph = request.app.state.graph
    snapshot = await graph.aget_state(thread_config(thread_id))
    tasks = cast(dict[str, Task], snapshot.values.get("tasks", {}))
    failed = {
        task.id: task.model_copy(update={"status": "pending", "error": None})
        for task in tasks.values()
        if task.status in STUCK
    }
    if failed:
        # Written as the node that precedes `dispatch`, so `dispatch` is the next step.
        await graph.aupdate_state(thread_config(thread_id), {"tasks": failed}, as_node="run_worker")
    elif not snapshot.next:
        raise HTTPException(status_code=409, detail="The run has nothing to retry")
    await store.set_status(request.app.state.sessions, thread_id, "running")
    supervisor.start(thread_id, None)
    return RunResponse(thread_id=thread_id, status="running")


@router.post("/{thread_id}/retry", status_code=202)
async def retry_run(thread_id: str, request: Request) -> RunResponse:
    """Run the failed tasks again, and the ones that were waiting for the user's decision.
    Finished tasks are not repeated. A run that used all its model calls needs `continue`."""
    run = await get_stored_run(request, thread_id)
    if run.status != "failed" or supervisor_of(request).is_running(thread_id):
        raise HTTPException(status_code=409, detail="Only a failed run can be retried")
    return await restart_stuck_tasks(request, thread_id)


@router.post("/{thread_id}/continue", status_code=202)
async def continue_run(thread_id: str, request: Request) -> RunResponse:
    """Allow a run that used all its model calls to use more, and carry on.

    A failed run runs its stuck tasks again, like `retry`. A finished run (a chat edit hit the
    cap) only gets the calls, so the next message works. Each call to this route adds
    the same allowance.
    """
    run = await get_stored_run(request, thread_id)
    if run.model_calls < run.call_limit:
        raise HTTPException(status_code=409, detail="The run still has model calls left")
    if supervisor_of(request).is_running(thread_id):
        raise HTTPException(status_code=409, detail="The run is still running")
    await store.allow_calls(request.app.state.sessions, thread_id, CALL_ALLOWANCE)
    if run.status == "failed":
        return await restart_stuck_tasks(request, thread_id)
    return RunResponse(thread_id=thread_id, status=run.status)


@router.post("/{thread_id}/resume", status_code=202)
async def resume_run(
    thread_id: str,
    decision: ReviewDecision,
    request: Request,
    idempotency_key: Annotated[str | None, Header(max_length=100)] = None,
) -> RunResponse:
    """Answer the brand review: approve, or ask for changes. The run carries on from where it
    paused.

    The run moves from `waiting` to `running` in one atomic update, so of two requests at once
    only the first resumes it and the other gets 409. A repeat of a request (same
    `Idempotency-Key`) after that is answered with the run as it stands, and does nothing.
    """
    run = await get_stored_run(request, thread_id)
    sessions = request.app.state.sessions
    if not await store.claim_resume(sessions, thread_id, idempotency_key):
        repeated = idempotency_key and idempotency_key == await store.last_resume_key(
            sessions, thread_id
        )
        if not repeated:
            raise HTTPException(status_code=409, detail="The run is not waiting for a decision")
        current = await get_stored_run(request, thread_id)
        return RunResponse(thread_id=thread_id, status=current.status)
    try:
        supervisor_of(request).resume(thread_id, decision)
    except Exception:
        await store.set_status(sessions, thread_id, run.status)  # nothing started, so still waiting
        raise
    return RunResponse(thread_id=thread_id, status="running")


@router.get("/{thread_id}/tasks/{task_id}")
async def get_task_report(thread_id: str, task_id: str, request: Request) -> TaskReport:
    """One task's saved result: its summary and the worker's full output."""
    report = await store.find_report(request.app.state.sessions, thread_id, task_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return report
