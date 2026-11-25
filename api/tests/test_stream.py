import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from support import BRIEF, api, approve, finish, parse_sse, run_to_the_end, start_run
from webkiln_agents.manager.context import Context
from webkiln_agents.testing import fake_context, fake_worker, fake_workers
from webkiln_agents.workers.base import WorkerInput, WorkerResult
from webkiln_api.streaming import RunChannel


def test_stream_sends_updates_and_worker_events_then_done() -> None:
    with api() as client:
        thread_id = start_run(client)
        paused = finish(client, thread_id)
        events = paused + approve(client, thread_id)

    # The first stream ends at the review, with the question in its last update.
    assert paused[-1] == ("done", {"status": "waiting"})
    assert any("__interrupt__" in data for name, data in paused if name == "updates")
    names = [name for name, _ in events]
    assert names[-1] == "done"
    assert events[-1][1] == {"status": "completed"}
    nodes = [next(iter(data)) for name, data in events if name == "updates"]
    assert nodes[:4] == ["process_initial_input", "seed_tasks", "analyze", "dispatch"]
    assert events[0][1] == {"process_initial_input": {"initial_input": BRIEF}}
    started = {
        d["agent"] for name, d in events if name == "custom" and d["event"] == "worker_started"
    }
    assert started == set(fake_workers())


def test_stream_of_a_finished_run_only_sends_done() -> None:
    with api() as client:
        thread_id = run_to_the_end(client)

        events = parse_sse(client.get(f"/runs/{thread_id}/stream").text)

    assert events == [("done", {"status": "completed"})]


def test_stream_of_an_unknown_run_is_404() -> None:
    with api() as client:
        assert client.get("/runs/nope/stream").status_code == 404


def test_a_second_connection_made_mid_run_gets_the_events_so_far_and_the_live_ones() -> None:
    gate = asyncio.Event()  # the seo worker waits here, so the run stays open

    async def waits_for_the_gate(packet: WorkerInput, context: Context) -> WorkerResult:
        if packet["task"]["agent"] == "seo":
            await gate.wait()
        return await fake_worker(packet, context)

    context = fake_context(workers=dict.fromkeys(fake_workers(), waits_for_the_gate))
    with api(context) as client, ThreadPoolExecutor() as pool:
        thread_id = start_run(client)
        first = pool.submit(finish, client, thread_id)
        started = threading.Event()

        def wait_until_seo_is_running() -> None:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                time.sleep(0.02)
                tasks = (client.get(f"/runs/{thread_id}").json()["state"] or {}).get("tasks", {})
                if tasks.get("seo", {}).get("status") == "running":
                    started.set()
                    return

        pool.submit(wait_until_seo_is_running).result(timeout=10)
        second = pool.submit(finish, client, thread_id)  # connects after the run started
        client.portal.call(gate.set)  # pyrefly: ignore[missing-attribute]

        first_events, second_events = first.result(timeout=10), second.result(timeout=10)

    assert started.is_set()
    assert second_events == first_events  # nothing missed, nothing doubled
    assert second_events[0][0] == "updates"


async def test_channel_replays_to_a_late_subscriber_and_ends_after_finish() -> None:
    channel = RunChannel()
    channel.publish("a")
    live = channel.subscribe()
    assert await anext(live) == "a"
    channel.publish("b")
    channel.finish()

    assert [event async for event in live] == ["b"]
    assert [event async for event in channel.subscribe()] == ["a", "b"]
