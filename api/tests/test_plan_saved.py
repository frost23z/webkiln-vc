from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.default_plan import default_plan
from webkiln_agents.manager.plan import Task
from webkiln_api.app import create_app
from webkiln_api.db import sqlalchemy_url
from webkiln_api.models import RunRecord
from webkiln_api.streaming import run_events
from webkiln_api.tables import Project, Run, TaskRow

BRIEF = {
    "brand_name": "Cedar Stay",
    "prompt": "A boutique hotel website with booking",
    "niche": "Hospitality",
}


def run_to_the_end(client: TestClient) -> str:
    thread_id = client.post("/runs", json=BRIEF).json()["thread_id"]
    client.get(f"/runs/{thread_id}/stream")
    return thread_id


def test_plan_is_saved_with_its_project_and_run() -> None:
    with TestClient(create_app()) as client:
        thread_id = run_to_the_end(client)

    engine = create_engine(sqlalchemy_url())
    try:
        with Session(engine) as session:
            run = session.scalars(select(Run).where(Run.thread_id == thread_id)).one()
            project = session.get(Project, run.project_id)
            tasks = session.scalars(select(TaskRow).where(TaskRow.run_id == run.id)).all()
    finally:
        engine.dispose()

    assert run.status == "completed"
    assert project is not None
    assert (project.brand_name, project.niche) == ("Cedar Stay", "Hospitality")
    planned = {t.id: t for t in default_plan().tasks}
    assert {t.id for t in tasks} == set(planned)
    assert all(t.status == "pending" for t in tasks)
    wireframe = next(t for t in tasks if t.id == "wireframe")
    assert wireframe.depends_on == planned["wireframe"].depends_on
    assert wireframe.agent == "wireframe"


def test_plan_rows_survive_a_new_app_instance() -> None:
    with TestClient(create_app()) as first:
        thread_id = run_to_the_end(first)

    # A new app has no memory of the run, so whatever it finds came from the database.
    async def saved_task_ids(app) -> set[str]:
        async with app.state.sessions() as session:
            run = (await session.scalars(select(Run).where(Run.thread_id == thread_id))).one()
            rows = await session.scalars(select(TaskRow).where(TaskRow.run_id == run.id))
            return {row.id for row in rows}

    second_app = create_app()
    with TestClient(second_app) as second:
        ids = second.portal.call(saved_task_ids, second_app)  # pyrefly: ignore[missing-attribute]

    assert ids == {t.id for t in default_plan().tasks}


def test_two_runs_save_two_projects_with_the_same_task_ids() -> None:
    with TestClient(create_app()) as client:
        first, second = run_to_the_end(client), run_to_the_end(client)

    engine = create_engine(sqlalchemy_url())
    try:
        with Session(engine) as session:
            runs = session.scalars(select(Run).where(Run.thread_id.in_([first, second]))).all()
            counts = [
                len(session.scalars(select(TaskRow).where(TaskRow.run_id == r.id)).all())
                for r in runs
            ]
    finally:
        engine.dispose()

    assert len(runs) == 2
    assert counts == [len(default_plan().tasks)] * 2


def test_a_run_that_fails_after_planning_is_saved_as_failed() -> None:
    class FailsAfterPlanning:
        async def astream(self, *args, **kwargs):
            yield {"type": "updates", "data": {"seed_tasks": {"tasks": {"niche": task}}}}
            raise RuntimeError("boom")

    task = Task(id="niche", agent="niche")
    thread_id = "failing-after-planning"
    record = RunRecord(brief=Input(**BRIEF), status="running")
    app = create_app()

    async def consume() -> list[str]:
        events = run_events(FailsAfterPlanning(), record, thread_id, app.state.sessions)  # pyrefly: ignore[bad-argument-type]
        return [event async for event in events]

    async def saved_status() -> str:
        async with app.state.sessions() as session:
            return (
                await session.scalars(select(Run.status).where(Run.thread_id == thread_id))
            ).one()

    with TestClient(app) as client:
        events = client.portal.call(consume)  # pyrefly: ignore[missing-attribute]
        status = client.portal.call(saved_status)  # pyrefly: ignore[missing-attribute]

    assert any("event: error" in event for event in events)
    assert status == "failed"
