import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from pydantic_core import to_jsonable_python

from webkiln_api.models import RunStatus


def sse(event: str, data: Any) -> str:
    """Format a single server-sent event, with JSON-encoded data"""
    return f"event: {event}\ndata: {json.dumps(to_jsonable_python(data, fallback=str))}\n\n"


class RunChannel:
    """The events of one running run, kept in memory.

    The run does not depend on any connection. A client that connects later, such as a page
    refresh or a second tab, first gets every event so far and then the live ones.
    """

    def __init__(self) -> None:
        self._events = list[str]()
        self._queues = set[asyncio.Queue[str | None]]()
        self._finished = False

    def publish(self, event: str) -> None:
        self._events.append(event)
        for queue in self._queues:
            queue.put_nowait(event)

    def finish(self) -> None:
        self._finished = True
        for queue in self._queues:
            queue.put_nowait(None)

    async def subscribe(self) -> AsyncGenerator[str]:
        queue = asyncio.Queue[str | None]()
        backlog = list(self._events)  # copied and registered with no await in between
        if self._finished:
            queue.put_nowait(None)
        else:
            self._queues.add(queue)
        try:
            for event in backlog:
                yield event
            while (event := await queue.get()) is not None:
                yield event
        finally:
            self._queues.discard(queue)


async def finished_events(status: RunStatus) -> AsyncGenerator[str]:
    """What a client gets when it connects to a run that is not running here."""
    yield sse("done", {"status": status})
