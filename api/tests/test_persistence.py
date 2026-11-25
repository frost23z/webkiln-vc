from sqlalchemy import select
from sqlalchemy.orm import Session
from support import BRIEF, api, run_to_the_end, saved_run, with_db
from webkiln_agents.manager.default_plan import default_plan
from webkiln_api.tables import Project, Run


def test_run_and_its_project_are_saved_when_the_run_is_created() -> None:
    with api() as client:
        thread_id = client.post("/runs", json=BRIEF).json()["thread_id"]
        # No waiting for the run: the rows exist from the moment the POST returned.
        run, _, _ = saved_run(thread_id)

        def read_project(session: Session) -> Project | None:
            return session.get(Project, run.project_id)

        project = with_db(read_project)

    assert project is not None
    assert (project.brand_name, project.niche) == ("Cedar Stay", "Hospitality")


def test_finished_run_has_its_tasks_and_outputs_in_postgres() -> None:
    with api() as client:
        thread_id = run_to_the_end(client)

    run, tasks, outputs = saved_run(thread_id)
    planned = {t.id: t for t in default_plan().tasks}
    assert run.status == "completed"
    assert {t.id for t in tasks} == set(planned)
    assert {t.status for t in tasks} == {"done"}
    assert (
        next(t for t in tasks if t.id == "wireframe").depends_on == planned["wireframe"].depends_on
    )
    assert {o.task_id for o in outputs} == set(planned)
    niche = next(o for o in outputs if o.task_id == "niche")
    assert (niche.summary, niche.data) == ("niche summary", {"agent": "niche"})


def test_task_report_endpoint_returns_the_saved_output() -> None:
    with api() as client:
        thread_id = run_to_the_end(client)

        report = client.get(f"/runs/{thread_id}/tasks/niche")
        missing = client.get(f"/runs/{thread_id}/tasks/nope")

    assert report.status_code == 200
    assert report.json() == {
        "task_id": "niche",
        "agent": "niche",
        "status": "done",
        "attempts": 1,
        "error": None,
        "summary": "niche summary",
        "output": {"agent": "niche"},
    }
    assert missing.status_code == 404


def test_run_survives_a_new_app_on_the_same_database() -> None:
    with api() as first:
        thread_id = run_to_the_end(first)
        before = first.get(f"/runs/{thread_id}").json()

    with api() as second:
        after = second.get(f"/runs/{thread_id}")
        events = second.get(f"/runs/{thread_id}/stream").text

    assert after.json() == before
    assert before["status"] == "completed"
    assert "event: done" in events
    assert '"status": "completed"' in events


def test_two_runs_save_two_projects_with_the_same_task_ids() -> None:
    with api() as client:
        first, second = run_to_the_end(client), run_to_the_end(client)

    def count(session: Session) -> int:
        return len(session.scalars(select(Run).where(Run.thread_id.in_([first, second]))).all())

    assert with_db(count) == 2
    assert len(saved_run(first)[1]) == len(saved_run(second)[1]) == len(default_plan().tasks)
