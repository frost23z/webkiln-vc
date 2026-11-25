import pytest
from pydantic import ValidationError
from webkiln_agents.manager.plan import Plan, PlannedTask, Task


def tasks(count: int) -> list[PlannedTask]:
    return [PlannedTask(id=f"t{i}", agent="niche") for i in range(count)]


def test_a_plan_needs_at_least_one_task() -> None:
    with pytest.raises(ValidationError):
        Plan(rationale="empty", tasks=[])


def test_a_plan_holds_at_most_twelve_tasks() -> None:
    assert len(Plan(rationale="full", tasks=tasks(12)).tasks) == 12
    with pytest.raises(ValidationError):
        Plan(rationale="too many", tasks=tasks(13))


def test_a_task_id_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        PlannedTask(id="", agent="niche")


def test_the_model_cannot_set_run_state() -> None:
    with pytest.raises(ValidationError):
        PlannedTask(id="niche", agent="niche", status="done")  # pyrefly: ignore[unexpected-keyword]


def test_a_runtime_task_starts_pending_with_no_attempts() -> None:
    planned = PlannedTask(id="brand", agent="brand", instruction="Be bold", depends_on=["niche"])
    task = Task.from_planned(planned)
    assert (task.status, task.attempts, task.error) == ("pending", 0, None)
    assert (task.id, task.instruction, task.depends_on) == ("brand", "Be bold", ["niche"])
