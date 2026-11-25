import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from pydantic_core import to_jsonable_python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.state import State

from webkiln_api.models import RunRecord, thread_config
from webkiln_api.plans import save_plan, save_run_status

logger = logging.getLogger(__name__)


def sse(event: str, data: Any) -> str:
    """Format a single server-sent event, with JSON-encoded data"""
    return f"event: {event}\ndata: {json.dumps(to_jsonable_python(data, fallback=str))}\n\n"


async def run_events(
    graph: CompiledStateGraph[State, None, Input, State],
    record: RunRecord,
    thread_id: str,
    sessions: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[str]:
    """Run the graph and yield server-sent events for each update, then a final `done` event."""
    try:
        async for chunk in graph.astream(
            record.brief.model_dump(),
            thread_config(thread_id),
            stream_mode=["updates", "custom"],
            version="v2",
        ):
            yield sse(chunk["type"], chunk["data"])
            if chunk["type"] == "updates" and "seed_tasks" in chunk["data"]:
                await save_planned_tasks(sessions, thread_id, record, chunk["data"]["seed_tasks"])
        record.status = "completed"
        await save_status(sessions, thread_id, record)
    except Exception:
        # The 200 is already sent, so the failure travels as an event. Keep details in the log.
        logger.exception("Run %s failed", thread_id)
        record.status = "failed"
        await save_status(sessions, thread_id, record)
        yield sse("error", {"code": "run_failed", "message": "The run failed."})
    finally:
        if record.status == "running":  # the client disconnected mid-run
            record.status = "failed"
    yield sse("done", {"status": record.status})


async def save_planned_tasks(
    sessions: async_sessionmaker[AsyncSession], thread_id: str, record: RunRecord, update: dict
) -> None:
    """Save the plan. A failure is logged and does not stop the run: the checkpoint has it too."""
    try:
        await save_plan(
            sessions, thread_id, record.brief, record.status, list(update["tasks"].values())
        )
    except Exception:
        logger.exception("Could not save the plan of run %s", thread_id)


async def save_status(
    sessions: async_sessionmaker[AsyncSession], thread_id: str, record: RunRecord
) -> None:
    try:
        await save_run_status(sessions, thread_id, record.status)
    except Exception:
        logger.exception("Could not save the status of run %s", thread_id)


async def finished_events(record: RunRecord) -> AsyncGenerator[str]:
    """What a client gets when it connects to a run that already ended."""
    yield sse("done", {"status": record.status})
