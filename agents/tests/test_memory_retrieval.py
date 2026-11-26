from typing import Any

from langgraph.store.memory import InMemoryStore
from samples import BRAND, NICHE
from test_approval import Session
from test_brand import brand_packet
from test_research import packet, text_of
from webkiln_agents.manager.context import Context
from webkiln_agents.manager.feedback import ReworkItem, ReworkOrder
from webkiln_agents.memory.embeddings import EMBEDDING_DIMS
from webkiln_agents.memory.service import Memory
from webkiln_agents.testing import FakeEmbeddings, FakeLLM
from webkiln_agents.workers.registry import WORKERS

NO_STOCK = "Never use stock photos on the site, show real photos of the place instead"
DARK = "Prefers a dark theme with gold accents on every site"
# The brand job and the manager rules mention "<memory>" in prose, so look for the block itself.
BLOCK = "<memory>\n"


def new_memory(user: str = "u1", project: str = "p1") -> tuple[Memory, FakeEmbeddings]:
    embeddings = FakeEmbeddings()
    store = InMemoryStore(index={"dims": EMBEDDING_DIMS, "embed": embeddings, "fields": ["text"]})
    return Memory(store, user, project), embeddings


async def test_the_brand_prompt_carries_the_users_preferences() -> None:
    memory, _ = new_memory()
    await memory.remember(NO_STOCK, "preference", "user_said")
    llm = FakeLLM({"brand": [BRAND]})

    await WORKERS["brand"](brand_packet(), Context(llm=llm, memory=memory))

    prompt = text_of(llm.prompts_for("brand")[0])
    assert BLOCK in prompt
    assert f"- (preference) {NO_STOCK}" in prompt
    assert "follow the user's lasting preferences" in prompt


async def test_the_memory_comes_before_the_managers_instruction() -> None:
    memory, _ = new_memory()
    await memory.remember(NO_STOCK, "preference", "user_said")
    llm = FakeLLM({"brand": [BRAND]})

    await WORKERS["brand"](
        brand_packet(instruction="Use dark green"), Context(llm=llm, memory=memory)
    )

    prompt = text_of(llm.prompts_for("brand")[0])
    assert prompt.index(BLOCK) < prompt.index("<instruction>\n")


async def test_a_brand_without_memory_has_no_memory_block() -> None:
    llm = FakeLLM({"brand": [BRAND]})
    await WORKERS["brand"](brand_packet(), Context(llm=llm))
    assert BLOCK not in text_of(llm.prompts_for("brand")[0])


async def test_an_empty_memory_adds_no_block() -> None:
    memory, _ = new_memory()
    llm = FakeLLM({"brand": [BRAND]})
    await WORKERS["brand"](brand_packet(), Context(llm=llm, memory=memory))
    assert BLOCK not in text_of(llm.prompts_for("brand")[0])


async def test_only_the_brand_reads_memory() -> None:
    memory, embeddings = new_memory()
    await memory.remember(NO_STOCK, "preference", "user_said")
    calls = embeddings.calls
    llm = FakeLLM({"niche": [NICHE]})

    await WORKERS["niche"](packet("niche"), Context(llm=llm, memory=memory))

    assert BLOCK not in text_of(llm.prompts_for("niche")[0])
    assert embeddings.calls == calls  # nothing was searched


async def test_a_broken_memory_does_not_stop_the_brand() -> None:
    class Broken(InMemoryStore):
        async def asearch(self, *args: Any, **kwargs: Any) -> Any:
            msg = "store down"
            raise RuntimeError(msg)

    memory = Memory(Broken(), "u1", "p1")
    llm = FakeLLM({"brand": [BRAND]})

    result = await WORKERS["brand"](brand_packet(), Context(llm=llm, memory=memory))

    assert result.summary == BRAND["summary"]
    assert BLOCK not in text_of(llm.prompts_for("brand")[0])


async def test_the_manager_reads_memory_when_it_plans_the_rework() -> None:
    memory, _ = new_memory()
    await memory.remember(DARK, "preference", "user_said")
    order = ReworkOrder(
        rationale="Brand only", tasks=[ReworkItem(task_id="brand", instruction="x")]
    )
    llm = FakeLLM({"manager:ReworkOrder": [order]})
    session = Session(llm)
    session.context.memory = memory

    await session.start()
    await session.resume({"action": "revise", "feedback": "Make the theme darker"})

    prompt = text_of(llm.prompts_for("manager", ReworkOrder)[0])
    assert BLOCK in prompt
    assert f"- (preference) {DARK}" in prompt
    assert "<feedback>\nMake the theme darker\n</feedback>" in prompt


async def test_the_managers_prompt_has_no_memory_block_without_memory() -> None:
    order = ReworkOrder(
        rationale="Brand only", tasks=[ReworkItem(task_id="brand", instruction="x")]
    )
    llm = FakeLLM({"manager:ReworkOrder": [order]})
    session = Session(llm)

    await session.start()
    await session.resume({"action": "revise", "feedback": "Make the theme darker"})

    assert BLOCK not in text_of(llm.prompts_for("manager", ReworkOrder)[0])
