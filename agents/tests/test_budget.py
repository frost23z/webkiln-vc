"""The cap on the model calls of one run."""

from collections import Counter
from typing import cast

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel
from test_edit_flow import HERO, build_files, intent, last_reply, new_words
from test_edit_flow import built as built_site
from webkiln_agents.budget import (
    CAP_REPLY,
    DEFAULT_CALL_LIMIT,
    BudgetedLLM,
    CallBudget,
    CallCapReached,
    cap_reached,
)
from webkiln_agents.llm import StructuredCall
from webkiln_agents.manager.context import Context, run_config
from webkiln_agents.manager.graph import build_graph
from webkiln_agents.testing import SAMPLE_BRIEF, fake_context, fake_worker, fake_workers
from webkiln_agents.workers.base import WorkerInput, WorkerResult
from webkiln_agents.workers.errors import is_transient


class Answer(BaseModel):
    text: str = "ok"


class CountingModel:
    """A provider that answers every `Answer` call and counts the calls that reach it."""

    def __init__(self) -> None:
        self.calls = Counter[str]()

    def structured[T: BaseModel](self, role: str, schema: type[T]) -> StructuredCall[T]:
        model = self

        class Call:
            async def ainvoke(self, input: list[BaseMessage]) -> T | None:
                model.calls[role] += 1
                # Only `Answer` is scripted. Any other schema gets "the model did not comply".
                return cast(T, Answer()) if schema is Answer else None

        return Call()


ASK: list[BaseMessage] = [HumanMessage("hi")]
BAD = ValueError("bad output")


def test_the_default_allowance_leaves_room_for_a_build_and_many_edits() -> None:
    assert DEFAULT_CALL_LIMIT == 60  # a build is 8 to 13 calls (plan 6.3)


def test_a_call_over_the_limit_raises_and_is_not_counted() -> None:
    budget = CallBudget(used=2, limit=3)
    budget.spend()
    assert (budget.used, budget.spent) == (3, True)
    with pytest.raises(CallCapReached, match="3 model calls"):
        budget.spend()
    assert budget.used == 3


def test_the_cap_is_never_retried() -> None:
    assert not is_transient(CallCapReached("used up"))


async def test_every_role_spends_from_one_budget_and_the_model_is_not_reached_after_the_cap() -> (
    None
):
    model, budget = CountingModel(), CallBudget(limit=3)
    llm = BudgetedLLM(model, budget)

    await llm.structured("manager", Answer).ainvoke(ASK)
    await llm.structured("brand", Answer).ainvoke(ASK)
    await llm.structured("content", Answer).ainvoke(ASK)
    with pytest.raises(CallCapReached):
        await llm.structured("manager", Answer).ainvoke(ASK)

    assert sum(model.calls.values()) == 3
    assert (budget.used, cap_reached(llm)) == (3, True)


def test_a_provider_that_is_not_budgeted_has_no_cap() -> None:
    assert cap_reached(None) is False
    assert cap_reached(CountingModel()) is False


async def test_workers_stop_at_the_cap_are_not_retried_and_ask_the_user() -> None:
    model, budget = CountingModel(), CallBudget(limit=3)  # one call goes to the brief analysis
    llm = BudgetedLLM(model, budget)

    async def worker(packet: WorkerInput, context: Context) -> WorkerResult:
        assert context.llm is not None
        await context.llm.structured(packet["task"]["agent"], Answer).ainvoke(ASK)
        return await fake_worker(packet, context)

    context = fake_context(workers=dict.fromkeys(fake_workers(), worker), llm=llm)
    result = await build_graph().ainvoke(SAMPLE_BRIEF, run_config(context), context=context)

    tasks = result["tasks"]
    research = [tasks[a] for a in ("niche", "audience", "competitor", "seo")]
    assert sorted(t.status for t in research) == ["done", "done", "needs_user", "needs_user"]
    assert all("model calls" in t.error for t in research if t.status == "needs_user")
    assert sum(model.calls.values()) == 3  # nothing past the cap reached the model
    assert tasks["brand"].status == "pending"  # it never started: its dependencies are not done
    assert all(t.attempts == 1 for t in research)  # the cap error was not retried


async def test_the_manager_failure_review_is_blocked_too_and_the_task_goes_to_the_user() -> None:
    llm = BudgetedLLM(CountingModel(), CallBudget(used=60, limit=60))

    async def worker(packet: WorkerInput, context: Context) -> WorkerResult:
        if packet["task"]["agent"] == "seo":
            raise BAD
        return await fake_worker(packet, context)

    context = fake_context(workers=dict.fromkeys(fake_workers(), worker), llm=llm)
    result = await build_graph().ainvoke(SAMPLE_BRIEF, run_config(context), context=context)

    assert result["tasks"]["seo"].status == "needs_user"  # no Manager answer: the user decides


async def test_a_chat_edit_at_the_cap_changes_nothing_and_says_why() -> None:
    session = await built_site(
        {"manager:EditIntent": [intent("content_edit", section=HERO)], "content:SectionCopy": []}
    )
    before = build_files(await session.state())
    session.context.llm = BudgetedLLM(session.llm, CallBudget(used=60, limit=60))

    state = await session.ask("Make the hero warmer")

    assert last_reply(state) == CAP_REPLY
    assert state.site is None  # no new version was made
    assert build_files(state) == before


async def test_a_chat_edit_stopped_by_the_cap_after_the_router_changes_nothing() -> None:
    words = new_words(HERO, "hero", "Wake up to the forest")
    session = await built_site(
        {
            "manager:EditIntent": [intent("content_edit", section=HERO)],
            "content:SectionCopy": [words],
        }
    )
    before = build_files(await session.state())
    # One call left: the router reads the message, then the section writer is refused.
    session.context.llm = BudgetedLLM(session.llm, CallBudget(used=59, limit=60))

    state = await session.ask("Make the hero warmer")

    assert last_reply(state) == CAP_REPLY
    assert state.site is None
    assert build_files(state) == before
