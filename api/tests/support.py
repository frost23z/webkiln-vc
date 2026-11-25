"""Helpers shared by the API tests."""

import json
from collections.abc import Callable, Generator
from contextlib import contextmanager

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from webkiln_agents.manager.context import Context
from webkiln_agents.testing import fake_context
from webkiln_api.app import create_app
from webkiln_api.db import sqlalchemy_url
from webkiln_api.tables import Run, TaskOutput, TaskRow

BRIEF = {
    "brand_name": "Cedar Stay",
    "prompt": "A boutique hotel website with booking",
    "niche": "Hospitality",
}


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        event, data = block.split("\n")
        events.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


@contextmanager
def api(context: Context | None = None) -> Generator[TestClient]:
    """A running API. Its workers are fakes unless `context` says otherwise."""
    with TestClient(create_app(context or fake_context())) as client:
        yield client


def start_run(client: TestClient, brief: dict | None = None) -> str:
    return client.post("/runs", json=brief or BRIEF).json()["thread_id"]


def finish(client: TestClient, thread_id: str) -> list[tuple[str, dict]]:
    """Follow the run to its end and return its events. The stream closes after `done`."""
    return parse_sse(client.get(f"/runs/{thread_id}/stream").text)


def approve(client: TestClient, thread_id: str, key: str | None = None) -> list[tuple[str, dict]]:
    """Approve the brand when the run is waiting, then follow the run to its end."""
    headers = {"Idempotency-Key": key} if key else {}
    response = client.post(f"/runs/{thread_id}/resume", json={"action": "approve"}, headers=headers)
    assert response.status_code == 202, response.text
    return finish(client, thread_id)


def run_to_review(client: TestClient, brief: dict | None = None) -> str:
    """Start a run and follow it until it pauses at the brand review."""
    thread_id = start_run(client, brief)
    assert finish(client, thread_id)[-1] == ("done", {"status": "waiting"})
    return thread_id


def run_to_the_end(client: TestClient, brief: dict | None = None) -> str:
    """A whole run, with the brand approved at the pause."""
    thread_id = run_to_review(client, brief)
    approve(client, thread_id)
    return thread_id


def with_db[T](read: Callable[[Session], T]) -> T:
    """Read the tables with a plain connection, the way another process would."""
    engine = create_engine(sqlalchemy_url())
    try:
        with Session(engine) as session:
            return read(session)
    finally:
        engine.dispose()


def saved_run(thread_id: str) -> tuple[Run, list[TaskRow], list[TaskOutput]]:
    def read(session: Session) -> tuple[Run, list[TaskRow], list[TaskOutput]]:
        run = session.scalars(select(Run).where(Run.thread_id == thread_id)).one()
        tasks = session.scalars(select(TaskRow).where(TaskRow.run_id == run.id)).all()
        outputs = session.scalars(select(TaskOutput).where(TaskOutput.run_id == run.id)).all()
        return run, list(tasks), list(outputs)

    return with_db(read)
