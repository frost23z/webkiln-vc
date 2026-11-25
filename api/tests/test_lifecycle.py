import asyncio
import time
from collections import Counter

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from support import BRIEF, api, finish, run_to_the_end, saved_run, start_run, with_db
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.context import Context
from webkiln_agents.testing import fake_context, fake_worker, fake_workers
from webkiln_agents.workers.base import WorkerInput, WorkerResult
from webkiln_api.tables import Project, Run

RESEARCH = {"audience", "competitor", "niche", "seo"}


def task_statuses(client: TestClient, thread_id: str) -> dict[str, str]:
    state = client.get(f"/runs/{thread_id}").json()["state"] or {}
    return {task_id: task["status"] for task_id, task in state.get("tasks", {}).items()}


def wait_for(client: TestClient, thread_id: str, done: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if list(task_statuses(client, thread_id).values()).count("done") >= done:
            return
        time.sleep(0.02)
    msg = f"fewer than {done} tasks finished"
    raise AssertionError(msg)


def test_a_failed_worker_fails_the_run_and_retry_repeats_only_that_worker() -> None:
    calls = Counter[str]()
    seo_works = False

    async def seo_fails_once(packet: WorkerInput, context: Context) -> WorkerResult:
        agent = packet["task"]["agent"]
        calls[agent] += 1
        if agent == "seo" and not seo_works:
            msg = "bad request"  # not a transient error, so the graph does not retry it
            raise ValueError(msg)
        return await fake_worker(packet, context)

    context = fake_context(workers=dict.fromkeys(fake_workers(), seo_fails_once))
    with api(context) as client:
        thread_id = start_run(client)
        events = finish(client, thread_id)
        assert events[-1] == ("done", {"status": "failed"})
        assert task_statuses(client, thread_id) == {
            "niche": "done",
            "audience": "done",
            "competitor": "done",
            "seo": "failed",
            "brand": "pending",
            "wireframe": "pending",
            "generate": "pending",
        }
        assert saved_run(thread_id)[0].status == "failed"

        seo_works = True
        retry = client.post(f"/runs/{thread_id}/retry")
        events = finish(client, thread_id)

        assert retry.status_code == 202
        assert events[-1] == ("done", {"status": "completed"})
        assert set(task_statuses(client, thread_id).values()) == {"done"}
        assert client.post(f"/runs/{thread_id}/retry").status_code == 409  # nothing failed now

    assert calls["seo"] == 2
    assert calls["niche"] == calls["audience"] == calls["competitor"] == 1


def test_retry_of_an_unknown_run_is_404() -> None:
    with api() as client:
        assert client.post("/runs/nope/retry").status_code == 404


def test_killing_the_api_mid_wave_and_restarting_runs_only_the_unfinished_workers() -> None:
    calls = Counter[str]()

    async def seo_hangs(packet: WorkerInput, context: Context) -> WorkerResult:
        calls[packet["task"]["agent"]] += 1
        if packet["task"]["agent"] == "seo":
            await asyncio.Event().wait()  # caught by the stop
        return await fake_worker(packet, context)

    async def counts(packet: WorkerInput, context: Context) -> WorkerResult:
        calls[packet["task"]["agent"]] += 1
        return await fake_worker(packet, context)

    first_context = fake_context(workers=dict.fromkeys(fake_workers(), seo_hangs))
    with api(first_context) as client:
        thread_id = start_run(client)
        wait_for(client, thread_id, done=3)
    # The first API is gone. Its run was cut off, so Postgres still says it is running.
    assert saved_run(thread_id)[0].status == "running"

    with api(fake_context(workers=dict.fromkeys(fake_workers(), counts))) as client:
        # The startup sweep already resumed the run. Following it ends when it is done.
        events = finish(client, thread_id)

        assert events[-1] == ("done", {"status": "completed"})
        assert set(task_statuses(client, thread_id).values()) == {"done"}

    run, tasks, outputs = saved_run(thread_id)
    assert run.status == "completed"
    assert {t.status for t in tasks} == {"done"}
    assert len(outputs) == len(tasks)
    assert {agent: calls[agent] for agent in RESEARCH} == {
        "audience": 1,
        "competitor": 1,
        "niche": 1,
        "seo": 2,  # started before the stop, started again after it
    }
    assert calls["brand"] == calls["wireframe"] == calls["generate"] == 1


def test_a_run_that_never_got_a_checkpoint_starts_again_from_its_brief() -> None:
    def create(session: Session) -> str:
        brief = Input(**BRIEF)
        project = Project(brand_name=brief.brand_name, niche=brief.niche, prompt=brief.prompt)
        session.add(project)
        session.flush()
        session.add(Run(project_id=project.id, thread_id="never-started", status="running"))
        session.commit()
        return "never-started"

    thread_id = with_db(create)

    with api() as client:
        events = finish(client, thread_id)

        assert events[-1] == ("done", {"status": "completed"})
        assert client.get(f"/runs/{thread_id}").json()["brief"] == BRIEF


def test_a_run_with_only_a_stale_status_is_corrected_by_the_sweep() -> None:
    with api() as client:
        thread_id = run_to_the_end(client)

    def make_stale(session: Session) -> None:
        run = session.query(Run).filter_by(thread_id=thread_id).one()
        run.status = "running"  # the process stopped right after the last step
        session.commit()

    with_db(make_stale)

    with api() as client:
        assert client.get(f"/runs/{thread_id}").json()["status"] == "completed"
