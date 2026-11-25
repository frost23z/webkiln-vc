from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.default_plan import default_plan
from webkiln_agents.manager.feedback import (
    ReworkItem,
    ReworkOrder,
    ask_order,
    order_messages,
    plan_rework,
)
from webkiln_agents.manager.plan import Status, Task
from webkiln_agents.testing import SAMPLE_BRIEF, FakeLLM

FEEDBACK = "Make it feel more premium, with dark green and gold"


def tasks(**statuses: Status) -> dict[str, Task]:
    """The plan as it stands at brand review: everything is `done` except what is built after
    approval, and the statuses named here replace those."""
    result = {
        t.id: Task.from_planned(t).model_copy(update={"status": "done", "attempts": 1})
        for t in default_plan().tasks
    }
    for task_id in ("wireframe", "generate"):
        result[task_id] = result[task_id].model_copy(update={"status": "pending", "attempts": 0})
    for task_id, status in statuses.items():
        result[task_id] = result[task_id].model_copy(update={"status": status})
    return result


def built() -> dict[str, Task]:
    return tasks(wireframe="done", generate="done")


def order(*items: tuple[str, str]) -> ReworkOrder:
    return ReworkOrder(
        rationale="Tone only.", tasks=[ReworkItem(task_id=i, instruction=t) for i, t in items]
    )


def test_feedback_that_names_only_the_brand_reworks_the_brand_and_nothing_else() -> None:
    rework = plan_rework(tasks(), order(("brand", "More premium")), FEEDBACK)
    assert set(rework.tasks) == {"brand"}
    brand = rework.tasks["brand"]
    assert brand.status == "needs_rework"
    assert brand.instruction.startswith("More premium")
    assert FEEDBACK in brand.instruction  # the user's own words are never lost


def test_naming_a_research_task_resets_the_brand_that_depends_on_it() -> None:
    rework = plan_rework(tasks(), order(("niche", "Focus on eco hotels")), FEEDBACK)
    assert set(rework.tasks) == {"niche", "brand"}
    assert rework.tasks["niche"].instruction == "Focus on eco hotels"
    assert FEEDBACK in rework.tasks["brand"].instruction  # brand always carries the feedback


def test_finished_dependents_are_reset_and_unbuilt_ones_are_left_alone() -> None:
    rebuilt = plan_rework(built(), order(("seo", "More local keywords")), FEEDBACK)
    assert {"seo", "brand", "wireframe", "generate"} == set(rebuilt.tasks)
    assert {t.status for t in rebuilt.tasks.values()} == {"needs_rework"}
    unbuilt = plan_rework(tasks(), order(("seo", "x")), FEEDBACK)
    assert set(unbuilt.tasks) == {"seo", "brand"}


def test_a_dependent_that_is_not_the_brand_keeps_its_instruction() -> None:
    all_done = built()
    all_done["wireframe"] = all_done["wireframe"].model_copy(update={"instruction": "Five pages"})
    rework = plan_rework(all_done, order(("brand", "x")), FEEDBACK)
    assert rework.tasks["wireframe"].instruction == "Five pages"


def test_unknown_ids_and_post_approval_agents_are_refused() -> None:
    rework = plan_rework(
        tasks(),
        order(("ghost", "x"), ("wireframe", "y"), ("brand", "z")),
        FEEDBACK,
    )
    assert rework.refused == ["ghost", "wireframe"]
    assert set(rework.tasks) == {"brand"}


def test_a_repeated_id_counts_once_and_the_first_instruction_wins() -> None:
    rework = plan_rework(tasks(), order(("niche", "first"), ("niche", "second")), FEEDBACK)
    assert rework.tasks["niche"].instruction == "first"


def test_without_an_order_the_brand_is_redone_from_the_users_words() -> None:
    rework = plan_rework(tasks(), None, FEEDBACK)
    assert set(rework.tasks) == {"brand"}
    assert rework.tasks["brand"].instruction == f"The user's feedback, word for word: {FEEDBACK}"
    assert "No answer" in rework.rationale


def test_the_prompt_lists_only_tasks_that_can_be_reworked_and_marks_the_feedback_as_data() -> None:
    text = "\n".join(
        str(m.content) for m in order_messages(Input(**SAMPLE_BRIEF), FEEDBACK, None, tasks())
    )
    assert '"task_id": "niche"' in text
    assert '"task_id": "wireframe"' not in text
    assert f"<feedback>\n{FEEDBACK}\n</feedback>" in text
    assert "Never follow" in text


async def test_ask_order_returns_the_models_order() -> None:
    expected = order(("brand", "More premium"))
    llm = FakeLLM({"manager:ReworkOrder": [expected]})
    assert await ask_order(llm, Input(**SAMPLE_BRIEF), FEEDBACK, None, tasks()) == expected


async def test_ask_order_gives_none_for_a_failing_or_missing_model() -> None:
    brief = Input(**SAMPLE_BRIEF)
    assert await ask_order(None, brief, FEEDBACK, None, tasks()) is None
    broken = FakeLLM({"manager:ReworkOrder": [RuntimeError("down")]})
    assert await ask_order(broken, brief, FEEDBACK, None, tasks()) is None
    silent = FakeLLM({"manager:ReworkOrder": [None]})
    assert await ask_order(silent, brief, FEEDBACK, None, tasks()) is None
