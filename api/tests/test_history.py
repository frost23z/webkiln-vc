from support import BRIEF, api, run_to_the_end


def test_history_lists_saved_runs_newest_first() -> None:
    with api() as client:
        older = run_to_the_end(client, {**BRIEF, "brand_name": "History Older"})
        newer = run_to_the_end(client, {**BRIEF, "brand_name": "History Newer"})

        runs = client.get("/runs").json()

    ids = [run["thread_id"] for run in runs]
    assert ids.index(newer) < ids.index(older)
    latest = next(run for run in runs if run["thread_id"] == newer)
    assert latest["status"] == "completed"
    assert latest["brief"]["brand_name"] == "History Newer"
    assert latest["updated_at"]


def test_history_is_there_after_a_restart() -> None:
    with api() as first:
        thread_id = run_to_the_end(first, {**BRIEF, "brand_name": "History Restart"})

    with api() as second:
        ids = [run["thread_id"] for run in second.get("/runs").json()]

    assert thread_id in ids


def test_history_respects_the_limit() -> None:
    with api() as client:
        run_to_the_end(client, {**BRIEF, "brand_name": "History One"})
        run_to_the_end(client, {**BRIEF, "brand_name": "History Two"})

        runs = client.get("/runs", params={"limit": 1}).json()

    assert len(runs) == 1


def test_history_rejects_a_bad_limit() -> None:
    with api() as client:
        assert client.get("/runs", params={"limit": 0}).status_code == 422
