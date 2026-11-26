"""Golden queries: for a fixed set of memories, each question must bring back the right record.

The embeddings are fake (a hash of the words, so a question has to share words with its record),
so these check the pipeline: scopes, kinds, the
current-only filter and the ranking. They say nothing about how good the real embedding model is;
`just check-memory` is the check for that.
"""

import pytest
from langgraph.store.base import IndexConfig
from langgraph.store.memory import InMemoryStore
from test_memory_retrieval import new_memory
from webkiln_agents.manager.remember import ARCHITECTURE as ARCHITECTURE_SLOT
from webkiln_agents.manager.remember import BRAND as BRAND_SLOT
from webkiln_agents.memory.embeddings import EMBEDDING_DIMS
from webkiln_agents.memory.records import Kind, Source, namespace_for
from webkiln_agents.memory.service import Memory
from webkiln_agents.testing import FakeEmbeddings

BRAND = (
    'Approved brand for Cedar Stay: tagline "Sleep among the cedars", calm warm tone, primary '
    "#2f5d50 on #f7f3ea, Playfair Display headings with Inter text. Positioning: quiet forest "
    "cabins for couples"
)
ARCHITECTURE = (
    "Site architecture for Cedar Stay: Home (hero, features, testimonials, cta); Rooms (hero, "
    "cabin grid, cta); Contact (hero, contact form). Menu: Home, Rooms, Contact"
)
HERO_DECISION = "Rewrote the hero section of the home page with a shorter heading"
FONT_DECISION = "Switched the headings to Lora after the user asked for a more classic look"
PALETTE_FEEDBACK = "The user said the palette felt too cold and asked for warmer greens"
CABINS = "Cedar Stay has eight wooden cabins and a sauna by Lake Nordic, open from May to October"
BOOKING = "Guests book by phone or email only, because there is no online payment"
KEYWORD = "Target keyword for the home page is forest cabin rental near Lake Nordic"
NO_STOCK = "Never use stock photos on the site, show real photos of the place instead"
SHORT_HEADINGS = "Always keep headings under five words"
DARK = "Prefers a dark theme with gold accents on every site"
OTHER_BRAND = (
    'Approved brand for Harbor Bakery: tagline "Bread at sunrise", primary #c2410c on #fff7ed, '
    "Merriweather headings with Lato text. Positioning: a neighbourhood bakery"
)

PROJECT_RECORDS: list[tuple[str, Kind, Source]] = [
    (BRAND, "brand", "approved"),
    (ARCHITECTURE, "architecture", "user_said"),
    (HERO_DECISION, "decision", "user_said"),
    (FONT_DECISION, "decision", "user_said"),
    (PALETTE_FEEDBACK, "feedback", "user_said"),
    (CABINS, "business_fact", "user_said"),
    (BOOKING, "business_fact", "user_said"),
    (KEYWORD, "seo", "approved"),
]
PREFERENCES = [NO_STOCK, SHORT_HEADINGS, DARK]


async def seeded() -> Memory:
    memory, _ = new_memory()
    slots = {BRAND: BRAND_SLOT, ARCHITECTURE: ARCHITECTURE_SLOT}  # the build keeps these in slots
    for text, kind, source in PROJECT_RECORDS:
        await memory.remember(text, kind, source, slot=slots.get(text))
    for text in PREFERENCES:
        await memory.remember(text, "preference", "user_said")
    return memory


async def top_project_record(memory: Memory, query: str) -> str:
    """The first line of the project's own records for `query` (the project comes first)."""
    found = await memory.recall(query, limit=3)
    return next(r.record.text for r in found if r.record.kind != "preference")


async def top_preference(memory: Memory, query: str) -> str:
    found = await memory.recall(query, limit=3)
    return next(r.record.text for r in found if r.record.kind == "preference")


PROJECT_QUERIES = [
    ("approved brand tagline and primary colour", BRAND),
    ("brand positioning and tone", BRAND),
    ("site architecture: pages and sections", ARCHITECTURE),
    ("site menu with Home, Rooms and Contact", ARCHITECTURE),
    ("the hero heading we rewrote on the home page", HERO_DECISION),
    ("why did we switch the headings to Lora", FONT_DECISION),
    ("the palette felt too cold, warmer greens", PALETTE_FEEDBACK),
    ("how many cabins are there and is there a sauna", CABINS),
    ("when is Lake Nordic open from May", CABINS),
    ("can guests pay online or book by phone", BOOKING),
    ("target keyword for the home page", KEYWORD),
    ("forest cabin rental near Lake Nordic keyword", KEYWORD),
]
PREFERENCE_QUERIES = [
    ("should I use stock photos or real photos of the place", NO_STOCK),
    ("never use stock photos", NO_STOCK),
    ("how long may headings be, under five words", SHORT_HEADINGS),
    ("dark theme with gold accents", DARK),
]


@pytest.mark.parametrize(("query", "expected"), PROJECT_QUERIES)
async def test_a_question_about_the_project_finds_its_record(query: str, expected: str) -> None:
    assert await top_project_record(await seeded(), query) == expected


@pytest.mark.parametrize(("query", "expected"), PREFERENCE_QUERIES)
async def test_a_question_about_a_habit_finds_the_users_preference(
    query: str, expected: str
) -> None:
    assert await top_preference(await seeded(), query) == expected


async def test_a_replaced_brand_is_never_recalled_but_its_history_is_kept() -> None:
    memory = await seeded()
    newer = BRAND.replace("#2f5d50", "#0b3d2e")
    await memory.remember(newer, "brand", "user_said", slot=BRAND_SLOT)
    await memory.remember(BRAND, "brand", "approved", slot=BRAND_SLOT)  # back to the first one
    found = [r.record for r in await memory.recall("approved brand primary colour", limit=5)]
    assert [r.text for r in found if r.kind == "brand"] == [BRAND]
    stored = await memory.store.asearch(namespace_for("u1", "p1", "brand"), limit=50)
    brands = [i.value for i in stored if i.value["kind"] == "brand"]
    assert sorted(b["status"] for b in brands) == ["current", "superseded", "superseded"]
    assert sorted(b["version"] for b in brands) == [1, 2, 3]  # every version stays in the store


async def test_another_projects_records_are_not_recalled_but_preferences_follow_the_user() -> None:
    store = InMemoryStore(index=new_store_index())
    first, second = Memory(store, "u1", "p1"), Memory(store, "u1", "p2")
    await first.remember(BRAND, "brand", "approved")
    await first.remember(NO_STOCK, "preference", "user_said")
    await second.remember(OTHER_BRAND, "brand", "approved")

    texts = [r.record.text for r in await second.recall("approved brand and photos", limit=3)]
    assert OTHER_BRAND in texts
    assert BRAND not in texts  # the other project's brand stays in its project
    assert NO_STOCK in texts  # the user's preference comes along


async def test_one_user_never_recalls_what_another_user_saved() -> None:
    store = InMemoryStore(index=new_store_index())
    await Memory(store, "u1", "p1").remember(BRAND, "brand", "approved")
    assert await Memory(store, "u2", "p1").recall("approved brand", limit=3) == []


async def test_a_question_about_nothing_saved_returns_few_records_and_never_an_error() -> None:
    memory = await seeded()
    found = await memory.recall("zebra quantum xylophone", limit=3)
    assert len(found) <= 6  # three from the project and three preferences at most
    assert await Memory(InMemoryStore(index=new_store_index()), "u1", "p1").recall("x") == []


def new_store_index() -> IndexConfig:
    return IndexConfig(dims=EMBEDDING_DIMS, embed=FakeEmbeddings(), fields=["text"])
