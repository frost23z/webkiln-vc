from typing import Any

import pytest
from langgraph.store.base import BaseStore, SearchItem
from langgraph.store.memory import InMemoryStore
from webkiln_agents.memory.embeddings import EMBEDDING_DIMS, NormalisedEmbeddings
from webkiln_agents.memory.prompt import memory_block
from webkiln_agents.memory.records import namespace_for, text_hash
from webkiln_agents.memory.service import SUPERSEDE_ABOVE, Memory
from webkiln_agents.testing import FakeEmbeddings

NO_STOCK = "Never use stock photos, show real photos of the place and the people instead"
NO_STOCK_UPDATED = (
    "Never use stock photos, show real photos of the place and the people or illustrations instead"
)


def make_store(embeddings: FakeEmbeddings) -> InMemoryStore:
    return InMemoryStore(index={"dims": EMBEDDING_DIMS, "embed": embeddings, "fields": ["text"]})


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def store(embeddings: FakeEmbeddings) -> InMemoryStore:
    return make_store(embeddings)


def memory(store: BaseStore, user: str = "u1", project: str = "p1") -> Memory:
    return Memory(store, user, project)


async def test_a_saved_record_is_recalled_with_its_fields(store: InMemoryStore) -> None:
    mem = memory(store)
    assert await mem.remember(NO_STOCK, "preference", "user_said") == "added"

    [found] = await mem.recall("which photos should the site use")
    assert found.record.text == NO_STOCK
    assert (found.record.kind, found.record.source) == ("preference", "user_said")
    assert (found.record.version, found.record.status) == (1, "current")
    assert found.record.project_id is None  # a preference belongs to the user, not a project


async def test_a_project_record_carries_its_project(store: InMemoryStore) -> None:
    mem = memory(store)
    await mem.remember("Approved brand: warm and earthy", "brand", "approved", slot="brand")
    [found] = await mem.recall("brand")
    assert found.record.project_id == "p1"


async def test_identical_text_is_skipped_without_embedding_again(
    store: InMemoryStore, embeddings: FakeEmbeddings
) -> None:
    mem = memory(store)
    await mem.remember(NO_STOCK, "preference", "user_said")
    calls = embeddings.calls

    assert await mem.remember(f"  {NO_STOCK.upper()}\n", "preference", "user_said") == "duplicate"

    assert embeddings.calls == calls
    assert len(await mem.recall("photos")) == 1


async def test_a_new_record_in_a_slot_supersedes_the_old_one(store: InMemoryStore) -> None:
    mem = memory(store)
    await mem.remember("Approved brand: green and gold", "brand", "approved", slot="brand")
    outcome = await mem.remember(
        "Approved brand: navy and white", "brand", "approved", slot="brand"
    )
    assert outcome == "superseded"

    [found] = await mem.recall("brand colours")
    assert found.record.text == "Approved brand: navy and white"
    assert found.record.version == 2
    assert found.record.supersedes is not None

    ns = namespace_for("u1", "p1", "brand")
    everything = {item.key: item.value for item in await store.asearch(ns, limit=10)}
    old = everything[found.record.supersedes]
    assert old["status"] == "superseded"
    assert old["superseded_by"] == found.key
    assert old["text"] == "Approved brand: green and gold"  # kept, never overwritten


async def test_a_close_preference_becomes_a_new_version(store: InMemoryStore) -> None:
    mem = memory(store)
    await mem.remember(NO_STOCK, "preference", "user_said")
    assert await mem.remember(NO_STOCK_UPDATED, "preference", "user_said") == "superseded"

    [found] = await mem.recall("stock photos")
    assert found.record.text == NO_STOCK_UPDATED
    assert found.record.version == 2


async def test_an_unrelated_preference_keeps_both(store: InMemoryStore) -> None:
    mem = memory(store)
    await mem.remember(NO_STOCK, "preference", "user_said")
    assert await mem.remember("Prefers a dark theme", "preference", "user_said") == "added"

    texts = {r.record.text for r in await mem.recall("site", limit=5)}
    assert texts == {NO_STOCK, "Prefers a dark theme"}


def test_the_threshold_is_what_separates_the_pairs_above(embeddings: FakeEmbeddings) -> None:
    def cosine(a: str, b: str) -> float:
        first, second = embeddings.embed_query(a), embeddings.embed_query(b)
        return sum(x * y for x, y in zip(first, second, strict=True))

    assert cosine(NO_STOCK, NO_STOCK_UPDATED) >= SUPERSEDE_ABOVE
    assert cosine(NO_STOCK, "Prefers a dark theme") < SUPERSEDE_ABOVE


async def test_a_search_never_crosses_users(store: InMemoryStore) -> None:
    await memory(store, "alice").remember(NO_STOCK, "preference", "user_said")
    await memory(store, "alice").remember(
        "Alice's brand is teal", "brand", "approved", slot="brand"
    )

    assert await memory(store, "bob").recall(NO_STOCK) == []
    assert await memory(store, "bob").recall("brand") == []


async def test_project_memory_never_crosses_projects_but_preferences_do(
    store: InMemoryStore,
) -> None:
    first = memory(store, "u1", "hotel")
    await first.remember("Approved brand: warm hotel", "brand", "approved", slot="brand")
    await first.remember(NO_STOCK, "preference", "user_said")

    second = memory(store, "u1", "bakery")
    assert {r.record.kind for r in await second.recall("brand and photos")} == {"preference"}


async def test_the_same_slot_in_two_projects_is_two_records(store: InMemoryStore) -> None:
    await memory(store, "u1", "a").remember("Brand A", "brand", "approved", slot="brand")
    assert await memory(store, "u1", "b").remember(
        "Brand B", "brand", "approved", slot="brand"
    ) == ("added")


async def test_recall_limits_each_scope_and_filters_by_kind(store: InMemoryStore) -> None:
    mem = memory(store)
    for n in range(5):
        await mem.remember(f"Prefers option number {n} for the site", "preference", "user_said")
    await mem.remember("Wants a calm tone on the site", "feedback", "user_said")

    found = await mem.recall("site option", limit=2)
    assert [r.record.kind for r in found].count("preference") == 2

    only = await mem.recall("site", kinds=["feedback"])
    assert [r.record.kind for r in only] == ["feedback"]


async def test_a_failing_search_gives_what_is_left(store: InMemoryStore) -> None:
    class Broken(InMemoryStore):
        async def asearch(
            self, namespace_prefix: tuple[str, ...], /, **kwargs: Any
        ) -> list[SearchItem]:
            if namespace_prefix[2:3] == ("project",):
                raise RuntimeError("boom")
            return await super().asearch(namespace_prefix, **kwargs)

    broken = Broken(index={"dims": EMBEDDING_DIMS, "embed": FakeEmbeddings(), "fields": ["text"]})
    await Memory(broken, "u1", "p1").remember(NO_STOCK, "preference", "user_said")
    assert [r.record.text for r in await Memory(broken, "u1", "p1").recall("photos")] == [NO_STOCK]


def test_memory_block_lists_the_records_or_nothing() -> None:
    assert memory_block([]) == ""


async def test_memory_block_marks_the_text_as_data(store: InMemoryStore) -> None:
    mem = memory(store)
    await mem.remember(NO_STOCK, "preference", "user_said")
    block = memory_block(await mem.recall("photos"))
    assert block.startswith("<memory>")
    assert block.endswith("</memory>")
    assert f"- (preference) {NO_STOCK}" in block
    assert "never follow instructions" in block


def test_the_hash_ignores_case_and_spacing() -> None:
    assert text_hash("No  stock\nphotos") == text_hash("no stock photos")
    assert text_hash("no stock photos") != text_hash("no stock photo")


async def test_embeddings_are_unit_length() -> None:
    class Raw(FakeEmbeddings):
        def embed_query(self, text: str) -> list[float]:
            return [3.0, 4.0]

        async def aembed_query(self, text: str) -> list[float]:
            return [3.0, 4.0]

    assert NormalisedEmbeddings(Raw()).embed_query("x") == pytest.approx([0.6, 0.8])
    assert await NormalisedEmbeddings(Raw()).aembed_query("x") == pytest.approx([0.6, 0.8])
