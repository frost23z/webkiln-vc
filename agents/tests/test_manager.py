import pytest
from pydantic import ValidationError
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.default_plan import default_plan
from webkiln_agents.manager.graph import graph
from webkiln_agents.manager.validation import validate_plan

VALID = {
    "brand_name": "Cedar Stay",
    "prompt": "A boutique hotel website with booking",
    "niche": "Hospitality",
}


async def run(brief: dict) -> dict:
    return await graph.ainvoke(brief)


async def test_valid_input_is_stored_in_state() -> None:
    result = await run(VALID)
    assert result["initial_input"] == Input(**VALID)


async def test_input_is_stripped_before_it_is_stored() -> None:
    result = await run({**VALID, "brand_name": "  Cedar Stay  "})
    assert result["initial_input"].brand_name == "Cedar Stay"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("brand_name", "x"),  # too short
        ("brand_name", "   "),  # blank after stripping
        ("prompt", "too short"),
        ("niche", "ab"),
        ("niche", "n" * 81),  # too long
    ],
)
async def test_invalid_input_raises_validation_error(field, value) -> None:
    with pytest.raises(ValidationError):
        await run({**VALID, field: value})


async def test_every_run_starts_from_the_default_plan_without_a_model() -> None:
    # No model is passed, so a node that called one would fail here.
    result = await run(VALID)
    assert result["plan"] == default_plan()


def test_the_default_plan_is_valid() -> None:
    assert validate_plan(default_plan()) == []


async def test_the_four_research_tasks_depend_on_nothing_so_they_can_run_together() -> None:
    result = await run(VALID)
    research = [result["tasks"][name] for name in ("niche", "audience", "competitor", "seo")]
    assert all(task.depends_on == [] for task in research)


async def test_tasks_are_seeded_one_pending_task_per_planned_task() -> None:
    result = await run(VALID)
    planned = default_plan().tasks
    assert list(result["tasks"]) == [t.id for t in planned]
    assert all(t.status == "pending" and t.attempts == 0 for t in result["tasks"].values())
    assert result["tasks"]["wireframe"].depends_on == ["brand"]
    assert result["tasks"]["brand"].depends_on == ["niche", "audience", "competitor", "seo"]
