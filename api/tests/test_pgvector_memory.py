"""Long-term memory on a real pgvector store, and the API that feeds it.

Embeddings are fakes (hashed words), so nothing is fetched. The store, its tables and its vector
index are the real ones. Every test uses its own user ids, because the store is not cleared.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.store.postgres.aio import AsyncPostgresStore
from support import BRIEF, finish, parse_sse, with_db
from webkiln_agents.manager.feedback import ReworkItem, ReworkOrder
from webkiln_agents.memory.records import MemoryRecord
from webkiln_agents.memory.service import Memory
from webkiln_agents.testing import (
    SAMPLE_BRAND,
    FakeEmbeddings,
    FakeLLM,
    fake_context,
    fake_workers,
)
from webkiln_agents.workers.registry import WORKERS
from webkiln_api.app import create_app
from webkiln_api.config import get_api_settings
from webkiln_api.memory import index_config
from webkiln_api.tables import Project

NO_STOCK = "Never use stock photos, show real photos of the place and the people instead"
NO_STOCK_UPDATED = (
    "Never use stock photos, show real photos of the place and the people or illustrations instead"
)


def user() -> str:
    return f"user-{uuid4().hex[:12]}"


@pytest.fixture
async def store() -> AsyncGenerator[AsyncPostgresStore]:
    async with AsyncPostgresStore.from_conn_string(
        get_api_settings().DB_URL, index=index_config(FakeEmbeddings())
    ) as pg_store:
        await pg_store.setup()
        yield pg_store


async def current(store: AsyncPostgresStore, user_id: str) -> list[MemoryRecord]:
    items = await store.asearch(("user", user_id), filter={"status": "current"}, limit=100)
    return [MemoryRecord.model_validate(item.value) for item in items]


# The store and its vector index


async def test_a_record_is_saved_and_found_by_meaning(store: AsyncPostgresStore) -> None:
    mem = Memory(store, user(), "hotel")
    assert await mem.remember(NO_STOCK, "preference", "user_said") == "added"

    [found] = await mem.recall("which photos should the site use")

    assert found.record.text == NO_STOCK
    assert found.record.version == 1


async def test_a_search_never_crosses_users(store: AsyncPostgresStore) -> None:
    alice, bob = user(), user()
    await Memory(store, alice, "hotel").remember(NO_STOCK, "preference", "user_said")

    assert await Memory(store, bob, "hotel").recall("stock photos") == []
    assert len(await Memory(store, alice, "hotel").recall("stock photos")) == 1


async def test_a_search_never_crosses_projects(store: AsyncPostgresStore) -> None:
    owner = user()
    await Memory(store, owner, "hotel").remember(
        "Cedar Stay is a boutique hotel with eleven rooms", "business_fact", "approved"
    )
    await Memory(store, owner, "bakery").remember(
        "Rye and Rise is a sourdough bakery with a street counter", "business_fact", "approved"
    )

    texts = [r.record.text for r in await Memory(store, owner, "bakery").recall("what we sell")]

    assert texts == ["Rye and Rise is a sourdough bakery with a street counter"]


async def test_a_preference_follows_the_user_into_another_project(
    store: AsyncPostgresStore,
) -> None:
    owner = user()
    await Memory(store, owner, "hotel").remember(NO_STOCK, "preference", "user_said")

    texts = [r.record.text for r in await Memory(store, owner, "bakery").recall("stock photos")]

    assert texts == [NO_STOCK]


async def test_the_same_text_is_saved_once(store: AsyncPostgresStore) -> None:
    mem = Memory(store, user(), "hotel")
    assert await mem.remember(NO_STOCK, "preference", "user_said") == "added"
    assert await mem.remember(NO_STOCK, "preference", "user_said") == "duplicate"


async def test_a_changed_fact_makes_a_new_version_and_keeps_the_old_one(
    store: AsyncPostgresStore,
) -> None:
    owner = user()
    mem = Memory(store, owner, "hotel")
    await mem.remember("Brand guideline: deep green and gold", "brand", "approved", slot="brand")
    outcome = await mem.remember(
        "Brand guideline: charcoal and copper", "brand", "approved", slot="brand"
    )

    assert outcome == "superseded"
    [now] = await current(store, owner)
    assert (now.text, now.version) == ("Brand guideline: charcoal and copper", 2)
    assert now.supersedes is not None
    everything = await store.asearch(("user", owner), limit=100)
    old = [MemoryRecord.model_validate(i.value) for i in everything if i.key == now.supersedes]
    assert [(r.status, r.text) for r in old] == [
        ("superseded", "Brand guideline: deep green and gold")
    ]


async def test_a_query_returns_only_current_versions(store: AsyncPostgresStore) -> None:
    owner = user()
    mem = Memory(store, owner, "hotel")
    await mem.remember(NO_STOCK, "preference", "user_said")
    assert await mem.remember(NO_STOCK_UPDATED, "preference", "user_said") == "superseded"

    texts = [r.record.text for r in await mem.recall("stock photos")]

    assert texts == [NO_STOCK_UPDATED]


async def test_a_different_preference_is_kept_beside_the_first(
    store: AsyncPostgresStore,
) -> None:
    owner = user()
    mem = Memory(store, owner, "hotel")
    await mem.remember(NO_STOCK, "preference", "user_said")
    outcome = await mem.remember("Keep every heading under six words", "preference", "user_said")

    assert outcome == "added"
    assert len(await current(store, owner)) == 2


# The API: the user id, the per-run memory and the whole flow


def revise_order() -> ReworkOrder:
    return ReworkOrder(
        rationale="Only the look changes.",
        tasks=[ReworkItem(task_id="brand", instruction="No stock photos")],
        preferences=[NO_STOCK],
    )


@contextmanager
def memory_api(llm: FakeLLM) -> Generator[TestClient]:
    """An API with memory on, the real brand worker and fakes for the other agents."""
    context = fake_context(workers={**fake_workers(), "brand": WORKERS["brand"]}, llm=llm)
    with TestClient(create_app(context, FakeEmbeddings())) as client:
        yield client


def run_as(client: TestClient, user_id: str | None, *answers: dict) -> tuple[str, list]:
    """Start a run for `user_id`, answer the review with each of `answers`, and return the thread
    and every event the run sent."""
    headers = {"X-User-Id": user_id} if user_id else {}
    thread_id = client.post("/runs", json=BRIEF, headers=headers).json()["thread_id"]
    events = finish(client, thread_id)
    for answer in answers:
        assert client.post(f"/runs/{thread_id}/resume", json=answer).status_code == 202
        events += finish(client, thread_id)
    return thread_id, events


def saved(events: list) -> list[dict]:
    return [d for e, d in events if e == "custom" and d.get("event") == "memory_saved"]


def brand_prompt(llm: FakeLLM, index: int = 0) -> str:
    return "\n".join(str(m.content) for m in llm.prompts_for("brand")[index])


def test_a_preference_from_one_session_shapes_the_next() -> None:
    owner = user()
    first = FakeLLM({"brand": [SAMPLE_BRAND], "manager:ReworkOrder": [revise_order()]})
    with memory_api(first) as client:
        _, events = run_as(
            client,
            owner,
            {"action": "revise", "feedback": "No stock photos please"},
            {"action": "approve"},
        )
    assert [s["stage"] for s in saved(events)] == ["feedback", "approval"]

    second = FakeLLM({"brand": [SAMPLE_BRAND]})
    with memory_api(second) as client:  # a new API process and a new project, the same user
        run_as(client, owner)

    assert f"- (preference) {NO_STOCK}" in brand_prompt(second)


def test_another_user_does_not_get_the_preference() -> None:
    first = FakeLLM({"brand": [SAMPLE_BRAND], "manager:ReworkOrder": [revise_order()]})
    with memory_api(first) as client:
        run_as(client, user(), {"action": "revise", "feedback": "No stock photos"})

    other = FakeLLM({"brand": [SAMPLE_BRAND]})
    with memory_api(other) as client:
        run_as(client, user())

    assert "<memory>\n" not in brand_prompt(other)


def test_a_run_without_a_user_has_no_memory() -> None:
    llm = FakeLLM({"brand": [SAMPLE_BRAND]})
    with memory_api(llm) as client:
        _, events = run_as(client, None, {"action": "approve"})

    assert saved(events) == []
    assert "<memory>\n" not in brand_prompt(llm)


def test_an_api_without_embeddings_runs_without_memory() -> None:
    """The default in tests (and with no Google key): the user id is kept, nothing is saved."""
    with TestClient(create_app(fake_context())) as client:
        owner = user()
        thread_id, events = run_as(client, owner, {"action": "approve"})

    assert saved(events) == []
    assert events[-1] == ("done", {"status": "completed"})
    assert with_db(lambda s: s.query(Project).filter(Project.user_id == owner).count()) == 1
    assert thread_id


def test_the_user_id_is_stored_on_the_project() -> None:
    owner = user()
    with memory_api(FakeLLM({"brand": [SAMPLE_BRAND]})) as client:
        run_as(client, owner)

    assert with_db(lambda s: s.query(Project).filter(Project.user_id == owner).count()) == 1


def test_a_run_resumed_after_a_restart_keeps_its_memory() -> None:
    """The user and project come from the saved run, not from the process that started it."""
    owner = user()
    first = FakeLLM({"brand": [SAMPLE_BRAND]})
    with memory_api(first) as client:
        thread_id, _ = run_as(client, owner)  # paused at the review

    with memory_api(FakeLLM({"brand": [SAMPLE_BRAND]})) as client:  # a new process
        resumed = client.post(f"/runs/{thread_id}/resume", json={"action": "approve"})
        assert resumed.status_code == 202
        events = parse_sse(client.get(f"/runs/{thread_id}/stream").text)

    assert [s["stage"] for s in saved(events)] == ["approval"]


@pytest.mark.parametrize("bad", ["has space", "dot.in.it", "a/b", "x" * 101])
def test_a_user_id_the_store_cannot_use_is_refused(bad: str) -> None:
    with TestClient(create_app(fake_context())) as client:
        response = client.post("/runs", json=BRIEF, headers={"X-User-Id": bad})

    assert response.status_code == 422


def test_the_browser_may_send_the_user_id() -> None:
    with TestClient(create_app(fake_context())) as client:
        response = client.options(
            "/runs",
            headers={
                "Origin": get_api_settings().UI_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-user-id,content-type",
            },
        )

    assert response.status_code == 200
    assert "x-user-id" in response.headers["access-control-allow-headers"].lower()


def test_a_restore_puts_the_restored_brand_back_in_memory() -> None:
    from test_editing import edit, edit_context, intent, say
    from webkiln_agents.testing import FakeEditLLM

    owner = user()
    script = {
        "manager:EditIntent": [intent("design_edit", "A darker green")],
        "brand:DesignChange": [{"primary": "#0b3d2e", "rationale": "A deeper green."}],
    }
    context = edit_context(script)
    context.llm = FakeEditLLM(script)
    with TestClient(create_app(context, FakeEmbeddings())) as client:
        headers = {"X-User-Id": owner}
        thread_id = client.post("/runs", json=BRIEF, headers=headers).json()["thread_id"]
        finish(client, thread_id)
        client.post(f"/runs/{thread_id}/resume", json={"action": "approve"})
        finish(client, thread_id)
        original = SAMPLE_BRAND["palette"]["primary"].lower()

        def brand_now() -> str:
            async def read() -> str:
                async with AsyncPostgresStore.from_conn_string(
                    get_api_settings().DB_URL, index=index_config(FakeEmbeddings())
                ) as pg:
                    items = await pg.asearch(
                        ("user", owner), filter={"slot": "brand", "status": "current"}
                    )
                    [item] = items
                    return item.value["text"]

            return asyncio.run(read())

        assert f"primary {original}" in brand_now()
        edit(client, thread_id, "Use a darker green", base_version=1)
        assert "primary #0b3d2e" in brand_now()  # the edit replaced the approved brand
        assert client.post(f"/runs/{thread_id}/versions/1/restore").status_code == 201
        assert f"primary {original}" in brand_now()  # and the restore put it back
        assert say(client, thread_id, "Hello").status_code == 202  # the run is not locked
