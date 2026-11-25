from webkiln_agents.manager.default_plan import default_plan
from webkiln_agents.manager.plan import Plan, PlannedTask, Task
from webkiln_api.checkpoint import serde


def roundtrip(value: object) -> object:
    return serde.loads_typed(serde.dumps_typed(value))


def test_plan_is_restored_as_a_plan() -> None:
    restored = roundtrip(default_plan())
    assert isinstance(restored, Plan)
    assert all(isinstance(task, PlannedTask) for task in restored.tasks)
    assert restored == default_plan()


def test_task_is_restored_as_a_task() -> None:
    task = Task(id="a", agent="niche", depends_on=["b"], status="done", attempts=2)
    restored = roundtrip(task)
    assert isinstance(restored, Task)
    assert restored == task
