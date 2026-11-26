from collections.abc import Mapping
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from samples import BRAND
from test_approval import Session
from test_research import text_of
from webkiln_agents.manager.context import run_config
from webkiln_agents.manager.feedback import ReworkItem, ReworkOrder
from webkiln_agents.manager.graph import build_graph
from webkiln_agents.manager.remember import brand_sentence, clip, edit_sentence
from webkiln_agents.memory.embeddings import EMBEDDING_DIMS
from webkiln_agents.memory.records import MAX_TEXT, MemoryRecord
from webkiln_agents.memory.service import Memory
from webkiln_agents.testing import SAMPLE_BRIEF, FakeEmbeddings, FakeLLM, fake_context, fake_workers
from webkiln_agents.workers.brand import BrandGuideline
from webkiln_agents.workers.registry import WORKERS

NO_STOCK = "Never use stock photos, show real photos of the place instead"
BLOCK = "<memory>\n"  # the prompts also mention "<memory>" in prose


def new_store() -> InMemoryStore:
    return InMemoryStore(
        index={"dims": EMBEDDING_DIMS, "embed": FakeEmbeddings(), "fields": ["text"]}
    )


async def records(store: BaseStore, user: str = "u1") -> list[MemoryRecord]:
    items = await store.asearch(("user", user), limit=100)
    return [MemoryRecord.model_validate(i.value) for i in items]


def order(*preferences: str) -> ReworkOrder:
    return ReworkOrder(
        rationale="Only the look changes.",
        tasks=[ReworkItem(task_id="brand", instruction="Go darker")],
        preferences=list(preferences),
    )


def session_with(memory: Memory | None, llm: FakeLLM | None = None) -> Session:
    session = Session(llm)
    session.context.memory = memory
    return session


async def events_of(session: Session, value: Any) -> list[Mapping[str, Any]]:
    """Resume the run and collect what it wrote to the custom stream."""
    stream = session.graph.astream(
        Command(resume=value), session.config, context=session.context, stream_mode="custom"
    )
    return [chunk async for chunk in stream]


async def test_approval_keeps_the_brand_and_the_research_it_was_built_on() -> None:
    store = new_store()
    session = session_with(Memory(store, "u1", "p1"))
    await session.start()

    await session.resume({"action": "approve"})

    by_kind = {r.kind: r for r in await records(store)}
    assert set(by_kind) == {"brand", "business_fact", "seo"}
    brand = by_kind["brand"]
    assert (brand.source, brand.slot, brand.project_id) == ("approved", "brand", "p1")
    assert "Cedar Stay" in brand.text
    assert BRAND["messaging"]["tagline"] in brand.text
    facts = [r for r in await records(store) if r.kind == "business_fact"]
    assert {r.slot for r in facts} == {"niche", "audience"}  # the competitor report is not kept
    assert all(r.status == "current" for r in await records(store))


async def test_approval_announces_what_it_saved() -> None:
    session = session_with(Memory(new_store(), "u1", "p1"))
    await session.start()

    seen = await events_of(session, {"action": "approve"})

    saved = [e for e in seen if e.get("event") == "memory_saved"]
    assert saved == [
        {
            "event": "memory_saved",
            "stage": "approval",
            "kinds": ["brand", "business_fact", "business_fact", "seo"],
        }
    ]


async def test_nothing_is_announced_or_saved_without_memory() -> None:
    session = session_with(None)
    await session.start()

    seen = await events_of(session, {"action": "approve"})

    assert [e for e in seen if e.get("event") == "memory_saved"] == []
    assert (await session.state())["brand_approved"] is True


async def test_hand_edits_are_kept_as_a_decision() -> None:
    store = new_store()
    session = session_with(Memory(store, "u1", "p1"))
    await session.start()
    await session.resume(
        {"action": "revise", "edits": {"accent": "#B8860B", "heading_font": "Lora"}}
    )

    await session.resume({"action": "approve"})

    [decision] = [r for r in await records(store) if r.kind == "decision"]
    assert decision.source == "user_said"
    assert "the accent colour to #b8860b" in decision.text
    assert "the heading font to Lora" in decision.text


async def test_the_brand_saved_is_the_one_that_was_approved_after_edits() -> None:
    store = new_store()
    session = session_with(Memory(store, "u1", "p1"))
    await session.start()
    await session.resume({"action": "revise", "edits": {"heading_font": "Lora"}})

    await session.resume({"action": "approve"})

    [brand] = [r for r in await records(store) if r.kind == "brand"]
    assert "Lora headings" in brand.text


async def test_feedback_keeps_the_users_words_and_a_lasting_preference() -> None:
    store = new_store()
    llm = FakeLLM({"manager:ReworkOrder": [order(NO_STOCK)]})
    session = session_with(Memory(store, "u1", "p1"), llm)
    await session.start()

    seen = await events_of(
        session, {"action": "revise", "feedback": "More premium, no stock photos"}
    )

    saved = {r.kind: r for r in await records(store)}
    assert set(saved) == {"feedback", "preference"}
    assert "More premium, no stock photos" in saved["feedback"].text
    assert saved["preference"].text == NO_STOCK
    assert saved["preference"].project_id is None
    assert {
        "event": "memory_saved",
        "stage": "feedback",
        "kinds": ["feedback", "preference"],
    } in seen


async def test_feedback_without_a_manager_answer_keeps_only_the_words() -> None:
    store = new_store()
    session = session_with(Memory(store, "u1", "p1"), FakeLLM({"manager:ReworkOrder": [None]}))
    await session.start()

    await session.resume({"action": "revise", "feedback": "Make it warmer"})

    assert [r.kind for r in await records(store)] == ["feedback"]


async def test_a_repeated_preference_is_not_saved_twice() -> None:
    store = new_store()
    llm = FakeLLM({"manager:ReworkOrder": [order(NO_STOCK), order(NO_STOCK)]})
    session = session_with(Memory(store, "u1", "p1"), llm)
    await session.start()

    await session.resume({"action": "revise", "feedback": "No stock photos"})
    seen = await events_of(session, {"action": "revise", "feedback": "Still no stock photos"})

    assert [r.kind for r in await records(store)].count("preference") == 1
    assert {"event": "memory_saved", "stage": "feedback", "kinds": ["feedback"]} in seen


async def test_a_failing_store_never_stops_the_run() -> None:
    class Broken(InMemoryStore):
        async def aput(self, *args: Any, **kwargs: Any) -> None:
            msg = "store down"
            raise RuntimeError(msg)

    session = session_with(Memory(Broken(), "u1", "p1"))
    await session.start()

    await session.resume({"action": "approve"})

    state = await session.state()
    assert state["brand_approved"] is True
    assert state["tasks"]["generate"].status == "done"


async def test_the_order_may_leave_the_preferences_out() -> None:
    parsed = ReworkOrder.model_validate({"rationale": "x", "tasks": []})
    assert parsed.preferences == []


def test_long_text_is_cut_to_what_a_record_allows() -> None:
    assert len(clip("word " * 500)) <= MAX_TEXT
    assert clip("  short   text ") == "short text"
    guideline = BrandGuideline.model_validate(BRAND)
    assert len(brand_sentence("Cedar Stay", guideline)) <= MAX_TEXT
    assert edit_sentence({"primary": "#112233"}) == (
        "The user set the primary colour to #112233 by hand in the brand guideline"
    )


async def run_flow(
    memory: Memory, llm: FakeLLM, thread: str, *answers: Mapping[str, Any]
) -> dict[str, Any]:
    """A run with the real brand worker, the other agents faked, and the given user answers."""
    workers = {**fake_workers(), "brand": WORKERS["brand"]}
    context = fake_context(workers=workers, llm=llm)
    context.memory = memory
    graph = build_graph(InMemorySaver())
    config = run_config(context, thread)
    await graph.ainvoke(dict(SAMPLE_BRIEF), config, context=context)
    for answer in answers:
        await graph.ainvoke(Command(resume=dict(answer)), config, context=context)
    return (await graph.aget_state(config)).values


async def test_a_preference_from_one_session_shapes_a_new_session() -> None:
    store = new_store()
    first_llm = FakeLLM({"brand": [BRAND], "manager:ReworkOrder": [order(NO_STOCK)]})
    await run_flow(
        Memory(store, "u1", "hotel"),
        first_llm,
        "one",
        {"action": "revise", "feedback": "No stock photos please"},
        {"action": "approve"},
    )

    second_llm = FakeLLM({"brand": [BRAND]})
    await run_flow(Memory(store, "u1", "bakery"), second_llm, "two")

    prompt = text_of(second_llm.prompts_for("brand")[0])
    assert f"- (preference) {NO_STOCK}" in prompt
    assert "Approved brand for Cedar Stay" not in prompt  # the other project's brand stays there
    assert "No stock photos please" not in prompt  # so does its feedback


async def test_another_user_never_sees_the_preference() -> None:
    store = new_store()
    await run_flow(
        Memory(store, "alice", "hotel"),
        FakeLLM({"brand": [BRAND], "manager:ReworkOrder": [order(NO_STOCK)]}),
        "one",
        {"action": "revise", "feedback": "No stock photos please"},
    )

    other = FakeLLM({"brand": [BRAND]})
    await run_flow(Memory(store, "bob", "hotel"), other, "two")

    assert BLOCK not in text_of(other.prompts_for("brand")[0])


async def test_the_same_project_remembers_its_own_history() -> None:
    store = new_store()
    await run_flow(
        Memory(store, "u1", "hotel"),
        FakeLLM({"brand": [BRAND], "manager:ReworkOrder": [order()]}),
        "one",
        {"action": "revise", "feedback": "Make the tone warmer"},
    )

    again = FakeLLM({"brand": [BRAND]})
    await run_flow(Memory(store, "u1", "hotel"), again, "two")

    prompt = text_of(again.prompts_for("brand")[0])
    assert "(feedback) Asked for changes to the brand guideline: Make the tone warmer" in prompt
