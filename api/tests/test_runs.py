from uuid import UUID

import pytest
from support import BRIEF, api, run_to_the_end, start_run


def test_valid_brief_starts_a_run_and_returns_its_thread_id() -> None:
    with api() as client:
        response = client.post("/runs", json=BRIEF)

    assert response.status_code == 201
    body = response.json()
    UUID(body["thread_id"])  # raises if it is not a uuid
    assert body["status"] == "running"


def test_each_run_gets_its_own_thread_id() -> None:
    with api() as client:
        assert start_run(client) != start_run(client)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("brand_name", "x"),
        ("brand_name", "   "),
        ("prompt", "too short"),
        ("niche", "n" * 81),
    ],
)
def test_invalid_brief_returns_422(field: str, value: str) -> None:
    with api() as client:
        response = client.post("/runs", json={**BRIEF, field: value})

    assert response.status_code == 422


def test_missing_field_returns_422() -> None:
    with api() as client:
        assert client.post("/runs", json={"brand_name": "Cedar Stay"}).status_code == 422


def test_get_run_returns_the_brief_and_the_graph_state() -> None:
    with api() as client:
        thread_id = run_to_the_end(client)

        body = client.get(f"/runs/{thread_id}").json()

    assert body["status"] == "completed"
    assert body["brief"] == BRIEF
    assert body["state"]["initial_input"] == BRIEF
    assert "brand_name" not in body["state"]  # input channels are not part of the declared state
    assert {task["status"] for task in body["state"]["tasks"].values()} == {"done"}


def test_get_unknown_run_returns_404() -> None:
    with api() as client:
        assert client.get("/runs/does-not-exist").status_code == 404
