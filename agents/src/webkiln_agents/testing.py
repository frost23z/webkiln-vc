"""Helpers shared by the tests of both packages. Nothing here calls a real model."""

import re
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from webkiln_agents.llm import LLMProvider
from webkiln_agents.manager.catalog import CATALOG
from webkiln_agents.manager.context import Context, run_config
from webkiln_agents.memory.embeddings import EMBEDDING_DIMS, unit
from webkiln_agents.workers.base import Worker, WorkerInput, WorkerResult

SAMPLE_BRIEF = {
    "brand_name": "Cedar Stay",
    "prompt": "A boutique hotel website with booking",
    "niche": "Hospitality",
}


class FakeStructured[T: BaseModel]:
    def __init__(self, llm: FakeLLM, role: str, schema: type[T]) -> None:
        self._llm = llm
        self._role = role
        self._schema = schema

    async def ainvoke(self, input: list[BaseMessage]) -> T | None:
        self._llm.prompts.append((self._role, list(input)))
        self._llm.schemas.append(self._schema)
        item = self._llm.next_response(self._role, self._schema)
        if isinstance(item, BaseException):
            raise item
        if item is None or isinstance(item, self._schema):
            return item
        return self._schema.model_validate(
            item
        )  # a bad dict raises ValidationError, as a model would


class FakeLLM:
    """A model that answers from a script, one list of responses per role.

    A response is a schema instance, a dict (validated against the schema, so a bad one raises
    `ValidationError`), `None` (a model that did not comply) or an exception (raised). When a
    role runs out of responses, its last one repeats. Every call is recorded in `prompts`.

    A script key is a role (`"manager"`) or a role and a schema name (`"manager:BriefAnalysis"`).
    The second form wins, so one role can answer calls with different schemas.
    """

    def __init__(self, script: Mapping[str, Sequence[Any]] | None = None) -> None:
        self._script = {role: list(items) for role, items in (script or {}).items()}
        self.prompts: list[tuple[str, list[BaseMessage]]] = []
        self.schemas: list[type[BaseModel]] = []  # the schema of each call, same order as prompts

    def next_response(self, role: str, schema: type[BaseModel] | None = None) -> Any:
        items = (self._script.get(f"{role}:{schema.__name__}") if schema else None) or (
            self._script.get(role)
        )
        if not items:
            msg = f"FakeLLM has no response for role {role!r}"
            raise AssertionError(msg)
        return items.pop(0) if len(items) > 1 else items[0]

    def structured[T: BaseModel](self, role: str, schema: type[T]) -> FakeStructured[T]:
        return FakeStructured(self, role, schema)

    def prompts_for(
        self, role: str, schema: type[BaseModel] | None = None
    ) -> list[list[BaseMessage]]:
        """The prompts sent to `role`, only those for `schema` when it is given."""
        return [
            messages
            for (name, messages), used in zip(self.prompts, self.schemas, strict=True)
            if name == role and (schema is None or used is schema)
        ]


async def fake_worker(packet: WorkerInput, _: Context) -> WorkerResult:
    agent = packet["task"]["agent"]
    return WorkerResult(summary=f"{agent} summary", data={"agent": agent})


def fake_workers() -> dict[str, Worker]:
    """A fake worker for every agent in the catalog."""
    return dict.fromkeys(CATALOG, fake_worker)


def fake_context(
    *,
    workers: Mapping[str, Worker] | None = None,
    llm: LLMProvider | None = None,
    max_concurrency: int = 4,
) -> Context:
    """A context where every agent has a fake worker, unless `workers` says otherwise."""
    return Context(
        workers=fake_workers() if workers is None else workers,
        llm=llm,
        max_concurrency=max_concurrency,
    )


APPROVE = {"action": "approve"}


async def run_approved(
    context: Context, brief: Mapping[str, Any] = SAMPLE_BRIEF, *, thread_id: str = "test"
) -> dict[str, Any]:
    """Run the graph to the end, approving the brand when the run pauses for review.

    Returns the final state values. The import is here because `graph` needs this module's peers.
    """
    from webkiln_agents.manager.graph import build_graph

    graph = build_graph(InMemorySaver())
    config = run_config(context, thread_id)
    await graph.ainvoke(dict(brief), config, context=context)
    while (await graph.aget_state(config)).interrupts:
        await graph.ainvoke(Command(resume=APPROVE), config, context=context)
    return (await graph.aget_state(config)).values


class FakeEmbeddings(Embeddings):
    """Embeddings from hashed words, so texts that share words are close and nothing is fetched.

    The cosine of two texts is the share of words they have in common, which makes similarity
    easy to set up in a test. `calls` counts the texts that were embedded.
    """

    def __init__(self, dims: int = EMBEDDING_DIMS) -> None:
        self.dims = dims
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        self.calls += 1
        vector = [0.0] * self.dims
        for word in set(re.findall(r"[a-z0-9#]+", text.casefold())):
            vector[zlib.crc32(word.encode()) % self.dims] += 1.0
        return unit(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
