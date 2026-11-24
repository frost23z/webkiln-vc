from fastapi.testclient import TestClient
from webkiln_api.app import create_app

BRIEF = {
    "brand_name": "Cedar Stay",
    "prompt": "A boutique hotel website with booking",
    "niche": "Hospitality",
}


def start_run(client: TestClient) -> str:
    thread_id = client.post("/runs", json=BRIEF).json()["thread_id"]
    client.get(f"/runs/{thread_id}/stream")  # the stream route starts the run
    return thread_id


def test_run_survives_a_new_app_on_the_same_database() -> None:
    with TestClient(create_app()) as first:
        thread_id = start_run(first)
        before = first.get(f"/runs/{thread_id}").json()

    # A new app has an empty in-memory run table, like an API process after a restart.
    with TestClient(create_app()) as second:
        after = second.get(f"/runs/{thread_id}")

    assert after.status_code == 200
    assert after.json() == before
    assert after.json()["status"] == "completed"
    assert after.json()["state"]["initial_input"] == BRIEF


def test_finished_run_reports_done_after_a_restart() -> None:
    with TestClient(create_app()) as first:
        thread_id = start_run(first)

    with TestClient(create_app()) as second:
        response = second.get(f"/runs/{thread_id}/stream")

    assert response.status_code == 200
    assert "event: done" in response.text
    assert '"status": "completed"' in response.text


def test_run_that_never_started_is_lost_after_a_restart() -> None:
    # Only the checkpoint is stored, and a run gets its first checkpoint when it starts.
    with TestClient(create_app()) as first:
        thread_id = first.post("/runs", json=BRIEF).json()["thread_id"]

    with TestClient(create_app()) as second:
        assert second.get(f"/runs/{thread_id}").status_code == 404
