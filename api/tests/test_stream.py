import json
from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.state import State
from webkiln_api.app import create_app

BRIEF = {
    "brand_name": "Cedar Stay",
    "prompt": "A boutique hotel website with booking",
    "niche": "Hospitality",
}


@pytest.fixture
def app() -> FastAPI:
    return create_app()


@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient]:
    with TestClient(app) as client:
        yield client


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        event, data = block.split("\n")
        events.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def start_run(client: TestClient) -> str:
    return client.post("/runs", json=BRIEF).json()["thread_id"]


def test_stream_sends_one_update_per_node_then_done(client: TestClient) -> None:
    thread_id = start_run(client)

    response = client.get(f"/runs/{thread_id}/stream")

    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["updates", "done"]
    assert events[0][1] == {"process_initial_input": {"initial_input": BRIEF}}
    assert events[1][1] == {"status": "completed"}


def test_run_state_is_available_after_the_stream(client: TestClient) -> None:
    thread_id = start_run(client)
    client.get(f"/runs/{thread_id}/stream")

    body = client.get(f"/runs/{thread_id}").json()

    assert body["status"] == "completed"
    assert body["state"]["initial_input"] == BRIEF


def test_stream_of_a_finished_run_only_sends_done(client: TestClient) -> None:
    thread_id = start_run(client)
    client.get(f"/runs/{thread_id}/stream")

    events = parse_sse(client.get(f"/runs/{thread_id}/stream").text)

    assert events == [("done", {"status": "completed"})]


def test_stream_of_a_running_run_is_refused(app: FastAPI, client: TestClient) -> None:
    thread_id = start_run(client)
    app.state.runs[thread_id].status = "running"

    assert client.get(f"/runs/{thread_id}/stream").status_code == 409


def test_stream_of_an_unknown_run_returns_404(client: TestClient) -> None:
    assert client.get("/runs/nope/stream").status_code == 404


def test_failing_node_becomes_an_error_event_without_details(
    app: FastAPI, client: TestClient
) -> None:
    def boom(state: Input) -> dict:
        msg = "secret internal detail"
        raise RuntimeError(msg)

    builder = StateGraph(State, input_schema=Input)
    builder.add_node("boom", boom)
    builder.add_edge(START, "boom")
    builder.add_edge("boom", END)
    app.state.graph = builder.compile(checkpointer=InMemorySaver())
    thread_id = start_run(client)

    response = client.get(f"/runs/{thread_id}/stream")

    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["error", "done"]
    assert events[0][1]["code"] == "run_failed"
    assert "secret internal detail" not in response.text
    assert events[1][1] == {"status": "failed"}
    assert client.get(f"/runs/{thread_id}").json()["status"] == "failed"
