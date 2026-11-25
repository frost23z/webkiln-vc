from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session
from support import api, approve, finish, run_to_review, start_run, with_db
from webkiln_agents.manager.context import Context
from webkiln_agents.manager.feedback import ReworkItem, ReworkOrder
from webkiln_agents.testing import FakeLLM, fake_context, fake_worker, fake_workers
from webkiln_agents.workers.base import WorkerInput, WorkerResult
from webkiln_api.tables import BrandVersion, Run


def counting() -> tuple[Context, Counter[str]]:
    calls = Counter[str]()

    async def worker(packet: WorkerInput, context: Context) -> WorkerResult:
        calls[packet["task"]["agent"]] += 1
        return await fake_worker(packet, context)

    return fake_context(workers=dict.fromkeys(fake_workers(), worker)), calls


def versions(thread_id: str) -> list[tuple[int, str, str | None]]:
    def read(session: Session) -> list[tuple[int, str, str | None]]:
        rows = (
            session.query(BrandVersion)
            .join(Run, Run.id == BrandVersion.run_id)
            .filter(Run.thread_id == thread_id)
            .order_by(BrandVersion.version)
        )
        return [(row.version, row.decision, row.feedback) for row in rows]

    return with_db(read)


def stored_status(thread_id: str) -> str:
    return with_db(lambda s: s.query(Run).filter_by(thread_id=thread_id).one().status)


def test_the_run_waits_at_the_brand_review_with_the_question_in_its_detail() -> None:
    context, calls = counting()
    with api(context) as client:
        thread_id = run_to_review(client)

        body = client.get(f"/runs/{thread_id}").json()
        assert body["status"] == "waiting"
        assert body["review"]["type"] == "brand_review"
        assert body["review"]["guideline"] == {"agent": "brand"}
        assert body["state"]["tasks"]["wireframe"]["status"] == "pending"
        assert client.get("/runs").json()[0]["status"] == "waiting"
        # Nothing to retry, and a stream of a waiting run only says so.
        assert client.post(f"/runs/{thread_id}/retry").status_code == 409
        assert finish(client, thread_id) == [("done", {"status": "waiting"})]
    assert stored_status(thread_id) == "waiting"
    assert calls["wireframe"] == calls["generate"] == 0
    assert versions(thread_id) == []


def test_after_a_restart_the_waiting_run_is_still_waiting_and_resumes_on_approve() -> None:
    first_context, _ = counting()
    with api(first_context) as client:
        thread_id = run_to_review(client)

    second_context, calls = counting()
    with api(second_context) as client:
        body = client.get(f"/runs/{thread_id}").json()
        # Not `failed`, and the startup sweep did not start it again.
        assert body["status"] == "waiting"
        assert body["review"]["type"] == "brand_review"
        assert not client.app.state.supervisor.is_running(thread_id)  # pyrefly: ignore[missing-attribute]
        assert not calls

        assert approve(client, thread_id)[-1] == ("done", {"status": "completed"})
        statuses = {
            t["status"] for t in client.get(f"/runs/{thread_id}").json()["state"]["tasks"].values()
        }
        assert statuses == {"done"}
    assert calls["wireframe"] == calls["generate"] == 1
    assert calls["niche"] == 0  # the research of the first process was not repeated
    assert versions(thread_id) == [(1, "approve", None)]


def test_a_run_left_at_running_while_it_waited_goes_back_to_waiting() -> None:
    with api(counting()[0]) as client:
        thread_id = run_to_review(client)

    def lose_the_answer(session: Session) -> None:
        session.query(Run).filter_by(thread_id=thread_id).one().status = "running"
        session.commit()

    with_db(lose_the_answer)

    with api(counting()[0]) as client:
        assert client.get(f"/runs/{thread_id}").json()["status"] == "waiting"
    assert stored_status(thread_id) == "waiting"


def test_two_approvals_at_once_resume_the_run_once() -> None:
    context, calls = counting()
    with api(context) as client:
        thread_id = run_to_review(client)

        def approve_once(key: str) -> int:
            return client.post(
                f"/runs/{thread_id}/resume",
                json={"action": "approve"},
                headers={"Idempotency-Key": key},
            ).status_code

        with ThreadPoolExecutor(2) as pool:
            codes = sorted(pool.map(approve_once, ["first", "second"]))
        assert codes == [202, 409]
        assert finish(client, thread_id)[-1] == ("done", {"status": "completed"})
    assert calls["wireframe"] == calls["generate"] == 1
    assert versions(thread_id) == [(1, "approve", None)]


def test_a_repeated_request_after_the_run_resumed_changes_nothing() -> None:
    context, calls = counting()
    with api(context) as client:
        thread_id = run_to_review(client)
        approve(client, thread_id, key="same")

        repeat = client.post(
            f"/runs/{thread_id}/resume",
            json={"action": "approve"},
            headers={"Idempotency-Key": "same"},
        )
        other = client.post(
            f"/runs/{thread_id}/resume",
            json={"action": "approve"},
            headers={"Idempotency-Key": "other"},
        )
        no_key = client.post(f"/runs/{thread_id}/resume", json={"action": "approve"})

        assert (repeat.status_code, repeat.json()["status"]) == (202, "completed")
        assert other.status_code == no_key.status_code == 409
    assert calls["wireframe"] == 1


def test_a_malformed_answer_is_refused_and_the_run_keeps_waiting() -> None:
    with api(counting()[0]) as client:
        thread_id = run_to_review(client)

        for bad in ({"action": "maybe"}, {"action": "revise"}, {}, {"action": "approve", "x": 1}):
            assert client.post(f"/runs/{thread_id}/resume", json=bad).status_code == 422, bad

        assert client.get(f"/runs/{thread_id}").json()["status"] == "waiting"
        assert client.post("/runs/unknown/resume", json={"action": "approve"}).status_code == 404


def test_feedback_reruns_only_the_brand_and_asks_for_a_new_review() -> None:
    context, calls = counting()
    order = ReworkOrder(
        rationale="Only the look changes.",
        tasks=[ReworkItem(task_id="brand", instruction="Darker")],
    )
    context.llm = FakeLLM({"manager:ReworkOrder": [order]})
    with api(context) as client:
        thread_id = run_to_review(client)

        response = client.post(
            f"/runs/{thread_id}/resume", json={"action": "revise", "feedback": "More premium"}
        )
        events = finish(client, thread_id)

        assert response.status_code == 202
        assert events[-1] == ("done", {"status": "waiting"})
        body = client.get(f"/runs/{thread_id}").json()
        assert body["review"]["version"] == 2
        assert body["review"]["revisions_left"] == 3
        attempts = {t: v["attempts"] for t, v in body["state"]["tasks"].items()}
        assert attempts["niche"] == attempts["audience"] == attempts["competitor"] == 1
        assert attempts["seo"] == 1
        assert attempts["brand"] == 2
        rework = [d for name, d in events if name == "custom" and d["event"] == "rework_planned"]
        assert rework[0]["tasks"] == ["brand"]
    assert versions(thread_id) == [(1, "revise", "More premium")]
    assert calls["brand"] == 2
    assert calls["niche"] == 1


def test_the_stream_of_a_resumed_run_ends_with_the_final_status() -> None:
    with api(counting()[0]) as client:
        thread_id = start_run(client)
        finish(client, thread_id)
        events = approve(client, thread_id)

        decisions = [
            d for name, d in events if name == "custom" and d["event"] == "review_decision"
        ]
        assert decisions == [{"event": "review_decision", "decision": "approve"}]
        assert events[-1] == ("done", {"status": "completed"})
