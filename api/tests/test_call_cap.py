"""The per-run cap on model calls, through the API: saved with the run, and lifted by the user."""

from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from support import api, finish, start_run
from webkiln_agents.llm import StructuredCall
from webkiln_agents.manager.context import Context
from webkiln_agents.testing import fake_context, fake_worker, fake_workers
from webkiln_agents.workers.base import WorkerInput, WorkerResult

CALLS_PER_WORKER = 20


class Answer(BaseModel):
    text: str = "ok"


class NoAnswerModel:
    """A provider whose model never complies: every call returns `None`."""

    def structured[T: BaseModel](self, role: str, schema: type[T]) -> StructuredCall[T]:
        class Call:
            async def ainvoke(self, input: list[BaseMessage]) -> T | None:
                return None

        return Call()


async def chatty(packet: WorkerInput, context: Context) -> WorkerResult:
    """A worker that makes many model calls, to use up an allowance of 60."""
    assert context.llm is not None
    call = context.llm.structured(packet["task"]["agent"], Answer)
    for _ in range(CALLS_PER_WORKER):
        await call.ainvoke([])
    return await fake_worker(packet, context)


def chatty_context() -> Context:
    return fake_context(workers=dict.fromkeys(fake_workers(), chatty), llm=NoAnswerModel())


def detail(client: Any, thread_id: str) -> dict:
    return client.get(f"/runs/{thread_id}").json()


def test_a_run_that_uses_its_allowance_stops_and_the_count_survives_a_restart() -> None:
    with api(chatty_context()) as client:
        thread_id = start_run(client)
        assert finish(client, thread_id)[-1] == ("done", {"status": "failed"})
        stopped = detail(client, thread_id)
        assert (stopped["model_calls"], stopped["call_limit"]) == (60, 60)
        tasks = stopped["state"]["tasks"]
        assert any(
            t["status"] == "needs_user" and "model calls" in t["error"] for t in tasks.values()
        )

    with api(chatty_context()) as client:  # a new process
        again = detail(client, thread_id)
        assert (again["model_calls"], again["status"]) == (60, "failed")


def test_continue_adds_an_allowance_and_runs_only_the_stuck_tasks() -> None:
    with api(chatty_context()) as client:
        thread_id = start_run(client)
        finish(client, thread_id)
        before = {k: t["attempts"] for k, t in detail(client, thread_id)["state"]["tasks"].items()}

        response = client.post(f"/runs/{thread_id}/continue")
        assert response.status_code == 202
        assert finish(client, thread_id)[-1] == ("done", {"status": "waiting"})

        after = detail(client, thread_id)
        assert after["call_limit"] == 120
        assert 60 < after["model_calls"] <= 120
        done_before = [k for k, t in before.items() if t == 1]
        tasks = after["state"]["tasks"]
        assert tasks["brand"]["status"] == "done"
        # a task that had finished before the stop was not run again
        finished = [k for k, a in before.items() if a == 1 and tasks[k]["attempts"] == 1]
        assert finished
        assert set(finished) <= set(done_before)


def test_continue_is_refused_while_the_run_still_has_calls() -> None:
    with api(fake_context()) as client:
        thread_id = start_run(client)
        finish(client, thread_id)
        response = client.post(f"/runs/{thread_id}/continue")
        assert response.status_code == 409
        assert detail(client, thread_id)["call_limit"] == 60


def test_continue_of_an_unknown_run_is_404() -> None:
    with api() as client:
        assert client.post("/runs/nope/continue").status_code == 404


def test_a_run_without_a_model_provider_has_no_cap_to_hit() -> None:
    with api(fake_context()) as client:
        thread_id = start_run(client)
        finish(client, thread_id)
        assert detail(client, thread_id)["model_calls"] == 0
