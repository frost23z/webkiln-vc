from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from webkiln_api.app import create_app

BRIEF = {
    "brand_name": "Cedar Stay",
    "prompt": "A boutique hotel website with booking",
    "niche": "Hospitality",
}


@pytest.fixture
def app_with_run() -> Generator[tuple[FastAPI, str]]:
    app = create_app()
    with TestClient(app) as client:
        yield app, client.post("/runs", json=BRIEF).json()["thread_id"]


def test_unknown_run_is_a_json_not_found_error() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/runs/nope")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found", "message": "Run not found"}}


def test_unknown_route_uses_the_same_error_shape() -> None:
    response = TestClient(create_app()).get("/nowhere")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_invalid_brief_lists_the_fields_without_echoing_values() -> None:
    response = TestClient(create_app()).post("/runs", json={**BRIEF, "brand_name": "x"})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["field"] == "brand_name"
    assert "at least 2 characters" in error["details"][0]["message"]
    assert '"input"' not in response.text


def test_running_run_gives_a_conflict_error(app_with_run: tuple[FastAPI, str]) -> None:
    app, thread_id = app_with_run
    app.state.runs[thread_id].status = "running"

    response = TestClient(app).get(f"/runs/{thread_id}/stream")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


def test_unexpected_exception_returns_json_without_a_traceback() -> None:
    app = create_app()

    @app.get("/boom")
    async def boom() -> None:
        msg = "secret internal detail"
        raise RuntimeError(msg)

    response = TestClient(app, raise_server_exceptions=False).get("/boom")

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "Internal server error."}
    }
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text
