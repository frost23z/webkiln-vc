"""Memory follows the site: what is built, what an edit changes, and what stays the same."""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.memory import InMemoryStore
from test_edit_flow import HERO, Session, intent, memory_session, new_words
from test_research import text_of
from webkiln_agents.manager.graph import build_graph
from webkiln_agents.manager.remember import ARCHITECTURE, BRAND, wireframe_sentence
from webkiln_agents.memory.records import MemoryRecord
from webkiln_agents.memory.service import Memory
from webkiln_agents.site.edits import EditIntent
from webkiln_agents.site.head import current_head
from webkiln_agents.testing import FakeEditLLM, sample_wireframe

PROJECT = ("user", "u1", "project", "p1")


async def records(store: InMemoryStore, slot: str | None = None) -> list[MemoryRecord]:
    items = await store.asearch(PROJECT, limit=100)
    found = [MemoryRecord.model_validate(item.value) for item in items]
    return [r for r in found if slot is None or r.slot == slot]


def current(found: list[MemoryRecord]) -> list[MemoryRecord]:
    return [r for r in found if r.status == "current"]


async def built_with_memory(script: dict[str, list[Any]]) -> tuple[Session, InMemoryStore]:
    session, store = memory_session(FakeEditLLM(script))
    await session.build()
    return session, store


async def test_the_architecture_is_kept_when_the_site_is_built() -> None:
    _, store = await built_with_memory({})
    [architecture] = current(await records(store, ARCHITECTURE))
    assert architecture.kind == "architecture"
    assert architecture.text == wireframe_sentence("Cedar Stay", sample_wireframe())
    for page in sample_wireframe().pages:
        assert page.name in architecture.text
    assert len(architecture.text) <= 600


async def test_the_brand_and_the_architecture_are_each_kept_once_after_a_build() -> None:
    _, store = await built_with_memory({})
    assert len(current(await records(store, BRAND))) == 1
    assert len(current(await records(store, ARCHITECTURE))) == 1


async def test_a_design_edit_replaces_the_approved_brand_instead_of_leaving_it_stale() -> None:
    session, store = await built_with_memory(
        {
            "manager:EditIntent": [intent("design_edit", "A darker green")],
            "brand:DesignChange": [{"primary": "#0b3d2e", "rationale": "A deeper green."}],
        }
    )
    [before] = current(await records(store, BRAND))
    assert "#0b3d2e" not in before.text
    await session.ask("Use a darker green")

    found = await records(store, BRAND)
    [now] = current(found)
    assert "primary #0b3d2e" in now.text
    assert now.version == before.version + 1
    assert now.supersedes is not None  # the old brand is kept as a version, not overwritten
    assert [r.status for r in found].count("superseded") == 1


async def test_a_structure_edit_updates_the_architecture() -> None:
    session, store = await built_with_memory(
        {"manager:EditIntent": [intent("remove_section", section="index-testimonials")]}
    )
    [before] = current(await records(store, ARCHITECTURE))
    assert "testimonials" in before.text.split("(")[1]
    await session.ask("Remove the testimonials from the homepage")

    [now] = current(await records(store, ARCHITECTURE))
    assert now.version == before.version + 1
    assert now.text != before.text
    assert now.text.count("testimonials") == before.text.count("testimonials") - 1


async def test_a_content_edit_leaves_the_brand_and_the_architecture_alone() -> None:
    session, store = await built_with_memory(
        {
            "manager:EditIntent": [intent("content_edit", "A warmer heading", section=HERO)],
            "content:SectionCopy": [new_words(HERO, "hero", "Wake up to the forest")],
        }
    )
    before = {r.slot: (r.version, r.text) for r in current(await records(store)) if r.slot}
    await session.ask("A warmer hero heading")
    after = {r.slot: (r.version, r.text) for r in current(await records(store)) if r.slot}
    assert after == before


async def test_an_edit_is_planned_with_the_approved_brand_and_the_architecture_in_view() -> None:
    """Scenario 9: the Manager and the writer both see what was approved, from memory."""
    session, _ = await built_with_memory(
        {
            "manager:EditIntent": [
                intent("content_edit", "Keep to the approved brand voice", section=HERO)
            ],
            "content:SectionCopy": [new_words(HERO, "hero", "Wake up to the forest")],
        }
    )
    await session.ask("Rewrite the hero heading so it matches our approved brand")
    manager = text_of(session.llm.prompts_for("manager", EditIntent)[0])
    assert "<memory>" in manager
    assert "(brand) Approved brand for Cedar Stay" in manager
    writer = text_of(session.llm.prompts_for("content", None)[0])
    assert "(brand) Approved brand for Cedar Stay" in writer


async def test_a_run_without_memory_builds_and_edits_as_before() -> None:
    session = await Session(
        FakeEditLLM(
            {"manager:EditIntent": [intent("remove_section", section="index-testimonials")]}
        )
    ).build()
    state = await session.ask("Remove the testimonials")
    assert state.site is not None
    assert state.site["version"] == 2


async def test_an_edit_after_a_restart_stays_consistent_with_the_approved_brand() -> None:
    """Scenario 9 across a restart: a new graph on the same checkpoints and a new `Memory` on the
    same store. The edit changes the primary colour and nothing else of the approved brand."""
    first, store = await built_with_memory({})
    state = await first.state()
    head = current_head(state.site, state.outputs)  # version 1: no edit has been saved yet
    assert head is not None
    palette, typography = head.brand["palette"], head.brand["typography"]

    script = {
        "manager:EditIntent": [intent("design_edit", "Make the primary colour a little darker")],
        "brand:DesignChange": [{"primary": "#0b3d2e", "rationale": "A deeper green."}],
    }
    restarted = Session(FakeEditLLM(script), Memory(store, "u1", "p1"))
    saver = first.graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    restarted.graph = build_graph(saver)  # the process came back
    state = await restarted.ask("Make the primary colour a little darker", base_version=1)

    assert "(brand) Approved brand for Cedar Stay" in text_of(
        restarted.llm.prompts_for("manager", EditIntent)[0]
    )
    assert state.site is not None
    assert state.site["version"] == 2
    new = state.site["brand"]["palette"]
    assert new["primary"] == "#0b3d2e"
    # The other colours are the approved ones. An edit writes colours in lower case.
    assert {k: v.lower() for k, v in new.items() if k != "primary"} == {
        k: v.lower() for k, v in palette.items() if k != "primary"
    }
    assert state.site["brand"]["typography"] == typography
    [now] = current(await records(store, BRAND))
    assert "primary #0b3d2e" in now.text  # and memory has followed the edit
