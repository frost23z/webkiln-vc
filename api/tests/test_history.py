from fastapi.testclient import TestClient
from webkiln_api.app import create_app

BRIEF = {
    "brand_name": "Cedar Stay",
    "prompt": "A boutique hotel website with booking",
    "niche": "Hospitality",
}


def start_run(client: TestClient, brand_name: str) -> str:
    thread_id = client.post("/runs", json={**BRIEF, "brand_name": brand_name}).json()["thread_id"]
    client.get(f"/runs/{thread_id}/stream")
    return thread_id


def test_history_lists_saved_runs_newest_first() -> None:
    with TestClient(create_app()) as client:
        older = start_run(client, "History Older")
        newer = start_run(client, "History Newer")

        runs = client.get("/runs").json()

    ids = [run["thread_id"] for run in runs]
    assert ids.index(newer) < ids.index(older)
    latest = next(run for run in runs if run["thread_id"] == newer)
    assert latest["status"] == "completed"
    assert latest["brief"]["brand_name"] == "History Newer"
    assert latest["updated_at"]


def test_history_is_there_after_a_restart() -> None:
    with TestClient(create_app()) as first:
        thread_id = start_run(first, "History Restart")

    with TestClient(create_app()) as second:
        ids = [run["thread_id"] for run in second.get("/runs").json()]

    assert thread_id in ids


def test_history_skips_runs_that_never_started() -> None:
    with TestClient(create_app()) as client:
        created = client.post("/runs", json=BRIEF).json()["thread_id"]

        ids = [run["thread_id"] for run in client.get("/runs").json()]

    assert created not in ids


def test_history_respects_the_limit() -> None:
    with TestClient(create_app()) as client:
        start_run(client, "History One")
        start_run(client, "History Two")

        runs = client.get("/runs", params={"limit": 1}).json()

    assert len(runs) == 1


def test_history_rejects_a_bad_limit() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/runs", params={"limit": 0}).status_code == 422
