from webkiln_agents.manager.analysis import BriefAnalysis
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


def test_brief_analysis_is_restored_as_a_brief_analysis() -> None:
    analysis = BriefAnalysis(
        purpose="Sell rooms.", business_model="Direct bookings.", expected_functionality=["Book"]
    )
    restored = roundtrip(analysis)
    assert isinstance(restored, BriefAnalysis)
    assert restored == analysis
