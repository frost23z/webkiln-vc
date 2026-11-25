from webkiln_agents.manager.default_plan import default_plan
from webkiln_agents.manager.plan import Plan, PlannedTask
from webkiln_agents.manager.validation import validate_plan


def make_plan(*tasks: PlannedTask) -> Plan:
    return Plan(rationale="test", tasks=list(tasks))


def full_tasks() -> list[PlannedTask]:
    """A valid plan's tasks, so a test can break exactly one thing."""
    return list(default_plan().tasks)


def test_default_plan_has_no_errors() -> None:
    assert validate_plan(default_plan()) == []


def test_duplicate_id_gives_one_error() -> None:
    tasks = [*full_tasks(), PlannedTask(id="niche", agent="seo")]
    assert validate_plan(make_plan(*tasks)) == ["task ids must be unique"]


def test_duplicate_id_does_not_hide_other_errors() -> None:
    bad = PlannedTask.model_construct(id="niche", agent="poet", depends_on=["ghost"])  # pyrefly: ignore[bad-argument-type]
    errors = validate_plan(make_plan(*full_tasks(), bad))
    assert errors == [
        "task ids must be unique",
        "niche uses unknown agent poet",
        "niche depends on unknown task ghost",
    ]


def test_unknown_agent_gives_one_error() -> None:
    # model_construct skips the Literal check, as a hand-built plan could
    bad = PlannedTask.model_construct(id="x", agent="poet", depends_on=[])  # pyrefly: ignore[bad-argument-type]

    assert validate_plan(make_plan(*full_tasks(), bad)) == ["x uses unknown agent poet"]


def test_unknown_dependency_gives_one_error() -> None:
    tasks = [
        *full_tasks(),
        PlannedTask(id="x", agent="niche", depends_on=["ghost"]),
    ]

    assert validate_plan(make_plan(*tasks)) == ["x depends on unknown task ghost"]


def test_duplicate_dependency_gives_one_error() -> None:
    tasks = [
        *full_tasks(),
        PlannedTask(
            id="x",
            agent="niche",
            depends_on=["niche", "niche"],
        ),
    ]

    assert validate_plan(make_plan(*tasks)) == ["x lists the same dependency more than once"]


def test_cycle_between_two_tasks_is_reported() -> None:
    tasks = [
        PlannedTask(id="niche", agent="niche", depends_on=["seo"]),
        PlannedTask(id="seo", agent="seo", depends_on=["niche"]),
        *[t for t in full_tasks() if t.agent not in ("niche", "seo")],
    ]

    errors = validate_plan(make_plan(*tasks))
    assert "dependency cycle: niche -> seo -> niche" in errors


def test_task_depending_on_itself_is_a_cycle() -> None:
    tasks = [
        PlannedTask(id="x", agent="niche", depends_on=["x"]),
        *full_tasks(),
    ]

    assert validate_plan(make_plan(*tasks)) == ["dependency cycle: x -> x"]


def test_shared_dependency_is_not_a_cycle() -> None:
    assert validate_plan(default_plan()) == []


def test_missing_agent_is_reported() -> None:
    tasks = [t for t in full_tasks() if t.agent != "seo"]

    for task in tasks:
        task.depends_on = [dep for dep in task.depends_on if dep != "seo"]

    assert validate_plan(make_plan(*tasks)) == ["plan is missing agent seo"]


def test_agent_without_required_dependency_is_reported() -> None:
    tasks = full_tasks()

    for task in tasks:
        if task.agent == "brand":
            task.depends_on = ["niche", "audience", "competitor"]

    assert validate_plan(make_plan(*tasks)) == ["brand (brand) must depend on a seo task"]


def test_wireframe_without_brand_is_reported() -> None:
    tasks = full_tasks()

    for task in tasks:
        if task.agent == "wireframe":
            task.depends_on = ["seo"]

    assert validate_plan(make_plan(*tasks)) == ["wireframe (wireframe) must depend on a brand task"]


def test_transitive_agent_dependency_is_valid() -> None:
    tasks = full_tasks()

    # Wireframe reaches brand through another niche task.
    tasks.append(
        PlannedTask(
            id="niche-extra",
            agent="niche",
            depends_on=["brand"],
        )
    )

    for task in tasks:
        if task.agent == "wireframe":
            task.depends_on = ["niche-extra"]

    assert validate_plan(make_plan(*tasks)) == []
