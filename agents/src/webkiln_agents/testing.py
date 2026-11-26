"""Helpers shared by the tests of both packages. Nothing here calls a real model."""

import asyncio
import re
import zlib
from collections import Counter
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
from webkiln_agents.site.vocabulary import rule_for
from webkiln_agents.workers.base import Worker, WorkerInput, WorkerResult
from webkiln_agents.workers.wireframe import Wireframe, WireframeDraft, finalize

SAMPLE_BRIEF = {
    "brand_name": "Cedar Stay",
    "prompt": "A boutique hotel website with booking",
    "niche": "Hospitality",
}


SAMPLE_BRAND: dict[str, Any] = {
    "summary": "Warm, calm boutique-hotel brand: deep green and gold, serif headings, local story.",
    "positioning": "A small owner-led hotel where design, calm and local character come first.",
    "personality_and_tone": {
        "traits": ["warm", "calm", "refined"],
        "voice": "Friendly and unhurried, like a host who knows the city well.",
        "do": ["Use short sentences", "Name real places"],
        "dont": ["Use hype words", "Shout in capitals"],
    },
    "messaging": {
        "tagline": "Stay like you live here",
        "value_proposition": "Book direct for the best rate, a late checkout and local tips.",
        "key_messages": ["Owner-led", "Book direct and save", "Rooms with a story"],
    },
    "palette": {
        "primary": "#1F4D3A",
        "secondary": "#C9A24B",
        "accent": "#8A5A44",
        "background": "#FAF7F0",
    },
    "typography": {
        "heading_font": "Playfair Display",
        "body_font": "Inter",
        "notes": "Playfair for headings and prices, Inter for everything else.",
    },
    "logo_direction": "A simple wordmark with a small cedar sprig, in the primary colour.",
    "image_guidelines": ["Natural light", "Real rooms, no stock", "Warm, muted tones"],
    "ui_ux_direction": ["Booking bar always visible", "Large photos", "Generous white space"],
    "inspiration": ["Aesop store pages", "Kinfolk magazine layouts"],
}


def _section(kind: str, title: str, brief: str, needs: list[str], **ctas: dict[str, str]) -> dict:
    return {
        "type": kind,
        "title": title,
        "content_brief": brief,
        "audience_needs": needs,
        **ctas,
    }


_BOOK = {"label": "Book direct", "target": "contact"}

SAMPLE_WIREFRAME_DRAFT: dict[str, Any] = {
    "summary": "Four pages: home, rooms, offers and contact, each answering a need of the guests.",
    "navigation": ["index", "rooms", "offers", "contact"],
    "functional_requirements": [
        {"kind": "form", "page": "contact", "description": "An enquiry form on the contact page."},
        {"kind": "map", "page": "contact", "description": "A map of the hotel's street."},
    ],
    "pages": [
        {
            "slug": "index",
            "name": "Home",
            "purpose": "Introduce Cedar Stay and send visitors to book.",
            "seo": {
                "target_keyword": "boutique hotel",
                "title": "Cedar Stay: a boutique hotel",
                "meta_description": "Cedar Stay is a small, owner-led boutique hotel. Book direct "
                "for the best rate and a late checkout.",
            },
            "sections": [
                _section(
                    "hero",
                    "Stay like you live here",
                    "A warm, calm welcome that says Cedar Stay is a boutique hotel with real local "
                    "character.",
                    ["Easy booking"],
                    cta_primary=_BOOK,
                ),
                _section(
                    "features",
                    "Why guests book direct",
                    "Three refined reasons to book direct: the best rate, a late checkout and "
                    "local tips.",
                    ["Flexible cancellation"],
                ),
                _section(
                    "testimonials",
                    "Guests say",
                    "Short, warm quotes from guests about the calm rooms and the hosts.",
                    ["Clear photos"],
                ),
                _section(
                    "cta_band",
                    "Ready to stay?",
                    "A calm closing invitation to book a room directly with the hotel.",
                    ["Easy booking"],
                    cta_primary=_BOOK,
                ),
            ],
        },
        {
            "slug": "rooms",
            "name": "Rooms",
            "purpose": "Show every room type with honest photos.",
            "seo": {
                "target_keyword": "boutique hotel rooms",
                "title": "Rooms at Cedar Stay",
                "meta_description": "See the boutique hotel rooms at Cedar Stay: real photos, "
                "clear details and a flexible cancellation policy.",
            },
            "sections": [
                _section(
                    "hero",
                    "Our rooms",
                    "An introduction to the rooms, with a refined, warm voice.",
                    ["Clear photos"],
                ),
                _section(
                    "gallery",
                    "Real rooms, real light",
                    "Three honest photos of the rooms in natural light.",
                    ["Clear photos"],
                ),
                _section(
                    "faq",
                    "Questions about the rooms",
                    "Answers about room sizes, beds and cancelling a stay.",
                    ["Flexible cancellation"],
                ),
            ],
        },
        {
            "slug": "offers",
            "name": "Offers",
            "purpose": "Present packages and rates for direct bookings.",
            "seo": {
                "target_keyword": "hotel packages",
                "title": "Hotel packages and offers",
                "meta_description": "Hotel packages from Cedar Stay: weekend escapes and long "
                "stays, with the best rate when you book direct.",
            },
            "sections": [
                _section(
                    "pricing",
                    "Packages",
                    "Two or three calm, clear packages with their prices.",
                    ["Easy booking", "Flexible cancellation"],
                    cta_primary=_BOOK,
                ),
            ],
        },
        {
            "slug": "contact",
            "name": "Contact",
            "purpose": "Let guests ask a question or book.",
            "seo": {
                "target_keyword": "contact boutique hotel",
                "title": "Contact Cedar Stay",
                "meta_description": "Contact the boutique hotel Cedar Stay to book a room or ask a "
                "question. We answer every message within a day.",
            },
            "sections": [
                _section(
                    "contact",
                    "Get in touch",
                    "A warm note, the address and opening hours, and an enquiry form.",
                    ["Easy booking"],
                ),
            ],
        },
    ],
}


def sample_copy(
    sections: Sequence[tuple[str, str, str]], keyword: str | None = None
) -> dict[str, Any]:
    """Valid copy for a page, from its `(section id, type, title)` triples, whatever the types.

    It holds exactly what each type's rule asks for, so the generate and API tests can share it.
    With a `keyword` (the page's target keyword), the first section's subheading uses it.
    """
    out = []
    for section_id, kind, title in sections:
        rule = rule_for(kind)
        count = rule.min_items if kind != "contact" else 2
        items = [
            {
                "title": f"{title} {n + 1}" if kind != "contact" else ["Address", "Hours"][n],
                "text": f"A clear sentence for item {n + 1} of {title}.",
                **({"meta": f"Meta {n + 1}"} if rule.meta else {}),
            }
            for n in range(count)
        ]
        out.append(
            {
                "id": section_id,
                "heading": title,
                "subheading": f"A short line under {title}."
                + (f" About {keyword}." if keyword and not out else ""),
                "body": "First paragraph of the copy.\n\nSecond paragraph of the copy."
                if rule.body
                else None,
                "items": items,
            }
        )
    return {"sections": out}


def sample_wireframe() -> Wireframe:
    """The sample wireframe as the wireframe worker saves it, with its section ids."""
    return finalize(WireframeDraft.model_validate(SAMPLE_WIREFRAME_DRAFT))


def sample_site_copy(wireframe: Wireframe | None = None) -> dict[str, dict[str, Any]]:
    """Valid copy for every page of a wireframe, by page slug. Each page uses its keyword."""
    frame = wireframe or sample_wireframe()
    return {
        page.slug: sample_copy(
            [(s.id, s.type, s.title) for s in page.sections], page.seo.target_keyword
        )
        for page in frame.pages
    }


def sample_image_plan(wireframe: Wireframe | None = None) -> dict[str, Any]:
    """A valid answer of the image worker: one note for every picture slot of the wireframe."""
    from webkiln_agents.site.images import slots_for

    frame = wireframe or sample_wireframe()
    slots = slots_for([(p.slug, [(s.id, s.type) for s in p.sections]) for p in frame.pages])
    return {
        "images": [
            {"slot": s.id, "prompt": f"Soft light for {s.id}.", "alt": f"A calm view for {s.id}"}
            for s in slots
        ]
    }


class FakeSiteLLM:
    """A model for the content and image roles. A page is answered by its slug in the prompt, so
    the order in which the parallel page calls arrive does not matter."""

    def __init__(
        self,
        *,
        copies: dict[str, Any] | None = None,
        images: Any = None,
        slow: set[str] | None = None,
        delays: Mapping[str, float] | None = None,
    ) -> None:
        self.copies = copies or sample_site_copy()
        self.images = images if images is not None else sample_image_plan()
        self.slow = slow or set()
        self.delays = delays or {}  # seconds a page waits before it is answered
        self.finished: list[str] = []  # page slugs in the order their calls were answered
        self.calls: Counter[str] = Counter()
        self.prompts: list[str] = []  # the text of every prompt, in the order the calls came
        self.answered: set[str] = set()
        self.others_done = asyncio.Event()  # every call but the slow ones has been answered

    def structured[T: BaseModel](self, role: str, schema: type[T]) -> _Call[T]:
        return _Call(self, role, schema)


class _Call[T: BaseModel]:
    def __init__(self, llm: FakeSiteLLM, role: str, schema: type[T]) -> None:
        self.llm, self.role, self.schema = llm, role, schema

    def _answered(self, name: str) -> None:
        self.llm.answered.add(name)
        if self.llm.answered >= {"index", "rooms", "contact", "images"}:
            self.llm.others_done.set()

    async def ainvoke(self, input: list[BaseMessage]) -> T | None:
        text = "\n".join(str(m.content) for m in input)
        self.llm.prompts.append(text)
        if self.role == "images":
            self.llm.calls["images"] += 1
            if isinstance(self.llm.images, Exception):
                raise self.llm.images
            answer = self.schema.model_validate(self.llm.images)
            self._answered("images")
            return answer
        slug = re.search(r'"slug": "([a-z0-9-]+)"', text)
        if slug is None:
            msg = "FakeSiteLLM expects every content prompt to name the page's slug"
            raise AssertionError(msg)
        self.llm.calls[slug[1]] += 1
        if slug[1] in self.llm.slow:
            await asyncio.Event().wait()
        await asyncio.sleep(self.llm.delays.get(slug[1], 0))
        answer = self.schema.model_validate(self.llm.copies[slug[1]])
        self.llm.finished.append(slug[1])
        self._answered(slug[1])
        return answer


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


EDIT_SCHEMAS = frozenset({"EditIntent", "SectionCopy", "DesignChange", "PageOrder"})


class FakeEditLLM:
    """A model for a whole session: a `FakeSiteLLM` builds the site, and a script answers the
    calls an edit makes (`manager:EditIntent`, `content:SectionCopy`, `brand:DesignChange` and
    `wireframe:PageOrder`). A new page is written by the site model, by its slug."""

    def __init__(
        self, script: Mapping[str, Sequence[Any]] | None = None, site: FakeSiteLLM | None = None
    ) -> None:
        self.site = site or FakeSiteLLM()
        self.edits = FakeLLM(script)

    def structured[T: BaseModel](self, role: str, schema: type[T]) -> Any:
        if schema.__name__ in EDIT_SCHEMAS:
            return self.edits.structured(role, schema)
        return self.site.structured(role, schema)

    def prompts_for(
        self, role: str, schema: type[BaseModel] | None = None
    ) -> list[list[BaseMessage]]:
        return self.edits.prompts_for(role, schema)


def site_context(llm: LLMProvider | None = None) -> Context:
    """Fake research, brand and wireframe workers and the real `generate` on `llm`."""
    from webkiln_agents.workers.generate import generate_worker

    async def brand(*_: Any) -> WorkerResult:
        return WorkerResult(summary="brand", data=SAMPLE_BRAND)

    async def wireframe(*_: Any) -> WorkerResult:
        return WorkerResult(summary="wireframe", data=sample_wireframe().model_dump(mode="json"))

    workers = {
        **fake_workers(),
        "brand": brand,
        "wireframe": wireframe,
        "generate": generate_worker,
    }
    return fake_context(workers=workers, llm=llm or FakeSiteLLM())


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
