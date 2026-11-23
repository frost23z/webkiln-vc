import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from pydantic_core import to_jsonable_python

from webkiln_api.models import RunRecord, thread_config

logger = logging.getLogger(__name__)


def sse(event: str, data: Any) -> str:
    """Format a single server-sent event, with JSON-encoded data"""
    return f"event: {event}\ndata: {json.dumps(to_jsonable_python(data, fallback=str))}\n\n"


async def run_events(
    graph: CompiledStateGraph, record: RunRecord, thread_id: str
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
        record.status = "completed"
    except Exception:
        # The 200 is already sent, so the failure travels as an event. Keep details in the log.
        logger.exception("Run %s failed", thread_id)
        record.status = "failed"
        yield sse("error", {"code": "run_failed", "message": "The run failed."})
    finally:
        if record.status == "running":  # the client disconnected mid-run
            record.status = "failed"
    yield sse("done", {"status": record.status})


async def finished_events(record: RunRecord) -> AsyncGenerator[str]:
    """What a client gets when it connects to a run that already ended."""
    yield sse("done", {"status": record.status})
