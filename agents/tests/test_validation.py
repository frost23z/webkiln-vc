from webkiln_agents.manager.plan import Plan, PlannedTask
from webkiln_agents.manager.validation import validate_plan


def make_plan(*tasks: PlannedTask) -> Plan:
    return Plan(rationale="test", tasks=list(tasks))


def test_plan_without_problems_has_no_errors() -> None:
    plan = make_plan(
        PlannedTask(id="a", agent="niche"), PlannedTask(id="b", agent="brand", depends_on=["a"])
    )
    assert validate_plan(plan) == []


def test_duplicate_id_gives_one_error() -> None:
    plan = make_plan(PlannedTask(id="a", agent="niche"), PlannedTask(id="a", agent="seo"))
    assert validate_plan(plan) == ["task ids must be unique"]


def test_unknown_agent_gives_one_error() -> None:
    # model_construct skips the Literal check, as a hand-built plan could
    bad = PlannedTask.model_construct(id="a", agent="poet", depends_on=[])  # pyrefly: ignore[bad-argument-type]
    assert validate_plan(make_plan(bad)) == ["a uses unknown agent poet"]


def test_unknown_dependency_gives_one_error() -> None:
    plan = make_plan(PlannedTask(id="a", agent="niche", depends_on=["ghost"]))
    assert validate_plan(plan) == ["a depends on unknown task ghost"]


def test_duplicate_dependency_gives_one_error() -> None:
    plan = make_plan(
        PlannedTask(id="a", agent="niche"),
        PlannedTask(id="b", agent="brand", depends_on=["a", "a"]),
    )
    assert validate_plan(plan) == ["b lists the same dependency more than once"]


def test_cycle_between_two_tasks_is_reported() -> None:
    plan = make_plan(
        PlannedTask(id="a", agent="niche", depends_on=["b"]),
        PlannedTask(id="b", agent="seo", depends_on=["a"]),
    )
    assert validate_plan(plan) == ["dependency cycle: a -> b -> a"]


def test_task_depending_on_itself_is_a_cycle() -> None:
    plan = make_plan(PlannedTask(id="a", agent="niche", depends_on=["a"]))
    assert validate_plan(plan) == ["dependency cycle: a -> a"]


def test_shared_dependency_is_not_a_cycle() -> None:
    plan = make_plan(
        PlannedTask(id="a", agent="niche"),
        PlannedTask(id="b", agent="seo", depends_on=["a"]),
        PlannedTask(id="c", agent="audience", depends_on=["a"]),
        PlannedTask(id="d", agent="brand", depends_on=["b", "c"]),
    )
    assert validate_plan(plan) == []
