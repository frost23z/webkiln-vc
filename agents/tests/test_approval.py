from collections import Counter
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from samples import BRAND
from webkiln_agents.manager.context import Context, run_config
from webkiln_agents.manager.feedback import ReworkItem, ReworkOrder
from webkiln_agents.manager.graph import build_graph
from webkiln_agents.manager.review import MAX_REVISIONS
from webkiln_agents.testing import (
    APPROVE,
    SAMPLE_BRIEF,
    FakeLLM,
    fake_context,
    fake_worker,
    fake_workers,
)
from webkiln_agents.workers.base import WorkerInput, WorkerResult

RESEARCH = ["audience", "competitor", "niche", "seo"]


class Session:
    """One run on one thread, driven the way the API drives it."""

    def __init__(self, llm: FakeLLM | None = None) -> None:
        self.calls = Counter[str]()
        self.packets: list[WorkerInput] = []
        self.llm = llm

        async def worker(packet: WorkerInput, context: Context) -> WorkerResult:
            agent = packet["task"]["agent"]
            self.calls[agent] += 1
            self.packets.append(packet)
            if agent == "brand":
                return WorkerResult(summary=BRAND["summary"], data=dict(BRAND))
            return await fake_worker(packet, context)

        self.context = fake_context(workers=dict.fromkeys(fake_workers(), worker), llm=llm)
        self.graph = build_graph(InMemorySaver())
        self.config = run_config(self.context, "gate")

    async def start(self) -> None:
        await self.graph.ainvoke(SAMPLE_BRIEF, self.config, context=self.context)

    async def resume(self, value: Any) -> None:
        await self.graph.ainvoke(Command(resume=value), self.config, context=self.context)

    async def state(self) -> dict[str, Any]:
        return (await self.graph.aget_state(self.config)).values

    async def payload(self) -> dict[str, Any] | None:
        interrupts = (await self.graph.aget_state(self.config)).interrupts
        return interrupts[0].value if interrupts else None


async def test_the_run_pauses_at_brand_review_with_the_wireframe_still_pending() -> None:
    session = Session()

    await session.start()

    state = await session.state()
    assert (await session.graph.aget_state(session.config)).next == ("brand_review",)
    assert state["tasks"]["brand"].status == "done"
    assert state["tasks"]["wireframe"].status == "pending"
    assert state["tasks"]["generate"].status == "pending"
    assert session.calls["wireframe"] == session.calls["generate"] == 0
    payload = await session.payload()
    assert payload is not None
    # The card gets the whole guideline, not the summary.
    assert payload["type"] == "brand_review"
    assert payload["guideline"] == BRAND
    assert payload["version"] == 1
    assert payload["revisions_left"] == MAX_REVISIONS
    assert payload["error"] is None


async def test_approve_resumes_the_same_thread_without_new_research_attempts() -> None:
    session = Session()
    await session.start()

    await session.resume(APPROVE)

    state = await session.state()
    assert await session.payload() is None
    assert {task.status for task in state["tasks"].values()} == {"done"}
    assert {state["tasks"][agent].attempts for agent in RESEARCH} == {1}
    assert state["brand_approved"] is True
    assert [entry["decision"] for entry in state["brand_reviews"]] == ["approve"]
    assert session.calls["wireframe"] == session.calls["generate"] == 1


async def test_feedback_that_names_only_the_brand_leaves_the_research_attempts_at_one() -> None:
    order = ReworkOrder(
        rationale="Only the look changes.",
        tasks=[ReworkItem(task_id="brand", instruction="Go darker")],
    )
    session = Session(FakeLLM({"manager:ReworkOrder": [order]}))
    await session.start()

    await session.resume({"action": "revise", "feedback": "Make it feel more premium"})

    state = await session.state()
    assert {state["tasks"][agent].attempts for agent in RESEARCH} == {1}
    assert state["tasks"]["brand"].attempts == 2
    assert session.calls["brand"] == 2
    assert all(session.calls[agent] == 1 for agent in RESEARCH)
    # The brand worker sees its last answer and the user's own words.
    second = [p for p in session.packets if p["task"]["agent"] == "brand"][1]
    assert second["previous"] == BRAND
    assert "Make it feel more premium" in second["task"]["instruction"]
    # A new card is shown, and the wireframe is still blocked.
    payload = await session.payload()
    assert payload is not None
    assert payload["version"] == 2
    assert payload["revisions_left"] == MAX_REVISIONS - 1
    assert state["tasks"]["wireframe"].status == "pending"
    assert [entry["decision"] for entry in state["brand_reviews"]] == ["revise"]
    assert state["brand_reviews"][0]["feedback"] == "Make it feel more premium"


async def test_feedback_that_changes_the_research_reruns_that_research_and_the_brand() -> None:
    order = ReworkOrder(
        rationale="New audience.",
        tasks=[ReworkItem(task_id="audience", instruction="Focus on business travellers")],
    )
    session = Session(FakeLLM({"manager:ReworkOrder": [order]}))
    await session.start()

    await session.resume({"action": "revise", "feedback": "We sell to business travellers"})

    assert session.calls["audience"] == 2
    assert session.calls["brand"] == 2
    assert session.calls["niche"] == session.calls["competitor"] == session.calls["seo"] == 1
    assert (await session.payload()) is not None


async def test_a_manager_that_gives_no_answer_still_reworks_the_brand() -> None:
    session = Session(FakeLLM({"manager:ReworkOrder": [None]}))
    await session.start()

    await session.resume({"action": "revise", "feedback": "Warmer, please"})

    assert session.calls["brand"] == 2
    assert all(session.calls[agent] == 1 for agent in RESEARCH)


async def test_a_malformed_resume_keeps_the_run_waiting() -> None:
    session = Session()
    await session.start()

    for bad in (
        {"action": "maybe"},
        {"action": "revise"},  # revise with nothing to change
        "approve",
        {"action": "approve", "extra": 1},
    ):
        await session.resume(bad)
        payload = await session.payload()
        assert payload is not None, bad
        assert payload["error"], bad
        assert payload["version"] == 1

    state = await session.state()
    assert state["tasks"]["wireframe"].status == "pending"
    assert state["brand_reviews"] == []

    await session.resume(APPROVE)  # the run still answers a good value
    assert await session.payload() is None
    assert (await session.state())["tasks"]["wireframe"].status == "done"


async def test_direct_edits_change_the_guideline_in_code_with_no_model_call() -> None:
    llm = FakeLLM()  # has no ReworkOrder response, so the test would notice a Manager call
    session = Session(llm)
    await session.start()

    await session.resume(
        {"action": "revise", "edits": {"accent": "#B8860B", "heading_font": "Lora"}}
    )

    state = await session.state()
    brand = state["outputs"]["brand"]
    assert brand["palette"]["accent"] == "#b8860b"
    assert brand["typography"]["heading_font"] == "Lora"
    assert "Lora" in state["summaries"]["brand"]  # the summary is rebuilt, not left stale
    assert llm.prompts_for("manager", ReworkOrder) == []
    assert session.calls["brand"] == 1
    payload = await session.payload()
    assert payload is not None
    assert payload["version"] == 2
    assert payload["guideline"] == brand
    assert state["brand_reviews"][0]["edits"] == {"accent": "#b8860b", "heading_font": "Lora"}


async def test_an_edit_that_breaks_the_contrast_rule_is_refused() -> None:
    session = Session()
    await session.start()

    await session.resume({"action": "revise", "edits": {"primary": BRAND["palette"]["background"]}})

    payload = await session.payload()
    assert payload is not None
    assert "contrast" in payload["error"]
    assert payload["guideline"] == BRAND
    assert (await session.state())["brand_reviews"] == []


async def test_the_fifth_round_of_changes_is_refused_and_approve_still_works() -> None:
    session = Session()
    await session.start()

    for round_number in range(MAX_REVISIONS):
        await session.resume(
            {"action": "revise", "edits": {"tagline": f"Tagline number {round_number}"}}
        )
        payload = await session.payload()
        assert payload is not None
        assert payload["error"] is None
        assert payload["revisions_left"] == MAX_REVISIONS - round_number - 1

    await session.resume({"action": "revise", "edits": {"tagline": "One round too many"}})

    payload = await session.payload()
    assert payload is not None
    assert "limit" in payload["error"]
    assert payload["guideline"]["messaging"]["tagline"] == f"Tagline number {MAX_REVISIONS - 1}"

    await session.resume(APPROVE)
    assert await session.payload() is None
    assert (await session.state())["brand_approved"] is True
