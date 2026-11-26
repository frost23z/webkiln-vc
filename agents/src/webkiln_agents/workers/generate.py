"""The generate worker: the files of the site, built by a compiled subgraph.

The subgraph runs inside the worker node of the parent graph, so the parent's retry, timeout and
failure handling cover it, and it checkpoints with the parent. Its shape:

    START -> write_page (one `Send` per page) --+
          -> write_images (one call, whole site) +--> assemble -> END

`write_page` calls the content worker once per page. `write_images` never fails the build: a site
without notes gets plain default alt text. `assemble` renders the files (plain code), runs the
site checks and stops, naming the files, when any check fails. A crash inside the subgraph
re-runs only the unfinished nodes of that subgraph. A retry of the whole task, started again by
`dispatch`, begins a new subgraph and writes every page again.
"""

from collections.abc import Callable
from typing import Annotated, Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from pydantic import BaseModel, Field

from webkiln_agents.manager.context import Context
from webkiln_agents.manager.scheduler import merge_dicts
from webkiln_agents.models import ConfigError
from webkiln_agents.site.checks import check_site
from webkiln_agents.site.copy import PageCopy
from webkiln_agents.site.images import ImageNote
from webkiln_agents.site.render import build_site
from webkiln_agents.workers.base import WorkerInput, WorkerResult
from webkiln_agents.workers.content import write_page
from webkiln_agents.workers.errors import WorkerOutputError
from webkiln_agents.workers.images import write_image_notes
from webkiln_agents.workers.wireframe import Wireframe

MAX_PROBLEMS_SHOWN = 6


class GenerateError(WorkerOutputError):
    """The site was built but did not pass the checks, or a page could not be written."""


class SiteBuild(BaseModel):
    """What the worker saves. The copy is kept so an edit can render again without parsing HTML."""

    files: dict[str, str]
    copies: dict[str, PageCopy]  # by page slug
    images: list[ImageNote]
    pages: list[str]  # page slugs, in wireframe order


class GenState(BaseModel):
    """The state of the subgraph. A page node also gets the `slug` of its page (`PageJob`)."""

    task_id: str
    instruction: str
    brief: dict[str, Any]
    analysis: dict[str, Any] | None
    brand: dict[str, Any]
    wireframe: dict[str, Any]
    # Page nodes run in parallel and each writes its own page, so the copies merge by slug.
    copies: Annotated[dict[str, dict[str, Any]], merge_dicts] = Field(default_factory=dict)
    notes: list[dict[str, Any]] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)


class PageJob(GenState):
    slug: str


def _emit(state: GenState, event: str, **extra: str) -> None:
    """A progress event for the stream. The wrapper passes it on to the parent's stream."""
    get_stream_writer()({"event": event, "task_id": state.task_id, "agent": "generate", **extra})


def _parent_writer() -> Callable[[Any], None]:
    """The parent's stream writer, or a writer that drops events when the worker is called on its
    own, outside any graph (as the tests of a worker do)."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _: None


def _llm(runtime: Runtime[Context]):
    llm = runtime.context.llm if runtime.context else None
    if llm is None:
        msg = "No model provider is configured for the generate agent"
        raise ConfigError(msg)
    return llm


def fan_out(state: GenState) -> list[Send]:
    """One page node per page and the image node, all in one step."""
    pages = Wireframe.model_validate(state.wireframe).pages
    sends = [Send("write_page", PageJob(**state.model_dump(), slug=page.slug)) for page in pages]
    return [*sends, Send("write_images", state)]


async def write_page_node(state: PageJob, runtime: Runtime[Context]) -> dict:
    wireframe = Wireframe.model_validate(state.wireframe)
    page = next(p for p in wireframe.pages if p.slug == state.slug)
    _emit(state, "page_started", page=page.slug)
    try:
        copy = await write_page(
            _llm(runtime),
            page,
            wireframe,
            state.brand,
            state.brief,
            state.analysis,
            state.instruction or None,
        )
    except WorkerOutputError as error:
        msg = f"The page {page.slug!r} could not be written: {error}"
        raise GenerateError(msg[:600]) from error
    _emit(state, "page_written", page=page.slug)
    return {"copies": {page.slug: copy.model_dump(mode="json")}}


async def write_images_node(state: GenState, runtime: Runtime[Context]) -> dict:
    wireframe = Wireframe.model_validate(state.wireframe)
    try:
        notes = await write_image_notes(_llm(runtime), wireframe, state.brand, state.brief)
    except WorkerOutputError:
        notes = []  # a missing note never fails the build: the render has a default for it
        _emit(state, "images_skipped")
    _emit(state, "images_written")
    return {"notes": [note.model_dump(mode="json") for note in notes]}


def assemble_node(state: GenState) -> dict:
    wireframe = Wireframe.model_validate(state.wireframe)
    copies = {slug: PageCopy.model_validate(copy) for slug, copy in state.copies.items()}
    notes = [ImageNote.model_validate(note) for note in state.notes]
    files = build_site(state.brand, state.brief["brand_name"], wireframe, copies, notes)
    if problems := check_site(files, wireframe):
        shown = "; ".join(problems[:MAX_PROBLEMS_SHOWN])
        more = len(problems) - MAX_PROBLEMS_SHOWN
        msg = f"The site failed its checks ({len(problems)}): {shown}"
        raise GenerateError((msg + (f"; and {more} more" if more > 0 else ""))[:600])
    _emit(state, "site_assembled", files=str(len(files)))
    return {"files": files}


def build_generate_graph() -> CompiledStateGraph[GenState, Context, GenState, GenState]:
    # No checkpointer of its own: inside the parent's node it uses the parent's.
    builder = StateGraph(GenState, context_schema=Context)
    builder.add_node("write_page", write_page_node)
    builder.add_node("write_images", write_images_node)
    builder.add_node("assemble", assemble_node)
    builder.add_conditional_edges(START, fan_out, ["write_page", "write_images"])
    builder.add_edge(["write_page", "write_images"], "assemble")
    builder.add_edge("assemble", END)
    return builder.compile()


GENERATE_GRAPH = build_generate_graph()


def initial_state(packet: WorkerInput) -> GenState:
    upstream = packet["upstream"]
    missing = [name for name in ("brand", "wireframe") if name not in upstream]
    if missing:
        msg = f"The generate agent needs the output of: {', '.join(missing)}"
        raise ConfigError(msg)
    task = packet["task"]
    return GenState(
        task_id=task["id"],
        instruction=task.get("instruction") or "",
        brief=packet["brief"],
        analysis=packet["analysis"],
        brand=upstream["brand"],
        wireframe=upstream["wireframe"],
    )


async def generate_worker(packet: WorkerInput, context: Context) -> WorkerResult:
    """Run the subgraph and pass its progress events on, because a subgraph's custom events do
    not reach the parent's stream by themselves."""
    writer = _parent_writer()
    final: Any = {}
    async for mode, chunk in GENERATE_GRAPH.astream(
        initial_state(packet), context=context, stream_mode=["custom", "values"]
    ):
        if mode == "custom":
            writer(chunk)
        else:
            final = chunk
    wireframe = Wireframe.model_validate(packet["upstream"]["wireframe"])
    build = SiteBuild(
        files=final["files"],
        copies={slug: PageCopy.model_validate(c) for slug, c in final["copies"].items()},
        images=[ImageNote.model_validate(n) for n in final["notes"]],
        pages=[p.slug for p in wireframe.pages],
    )
    summary = (
        f"Built {len(build.pages)} pages ({', '.join(build.pages)}) as {len(build.files)} files, "
        f"with notes for {len(build.images)} pictures."
    )
    return WorkerResult(summary=summary, data=build.model_dump(mode="json"))
