import asyncio
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_research import packet
from webkiln_agents.manager.context import Context, run_config
from webkiln_agents.site.checks import check_site
from webkiln_agents.site.copy import PageCopy
from webkiln_agents.site.images import ImagePlan, slots_for
from webkiln_agents.testing import (
    SAMPLE_BRAND,
    SAMPLE_BRIEF,
    FakeSiteLLM,
    fake_context,
    sample_image_plan,
    sample_site_copy,
    sample_wireframe,
)
from webkiln_agents.workers.base import WorkerInput
from webkiln_agents.workers.errors import WorkerOutputError
from webkiln_agents.workers.generate import (
    GENERATE_GRAPH,
    GenerateError,
    GenState,
    SiteBuild,
    assemble_node,
    generate_worker,
)
from webkiln_agents.workers.registry import WORKERS
from webkiln_agents.workers.wireframe import Wireframe

WIREFRAME = sample_wireframe()
COPIES = sample_site_copy(WIREFRAME)
SLOTS = slots_for([(p.slug, [(s.id, s.type) for s in p.sections]) for p in WIREFRAME.pages])
IMAGES = sample_image_plan(WIREFRAME)


def generate_packet(instruction: str = "") -> Any:
    base = packet("generate", instruction)
    base["upstream"] = {"brand": SAMPLE_BRAND, "wireframe": WIREFRAME.model_dump(mode="json")}
    return base


async def build(llm: FakeSiteLLM, **kw: str) -> SiteBuild:
    result = await generate_worker(generate_packet(**kw), Context(llm=llm))
    return SiteBuild.model_validate(result.data)


async def test_it_builds_every_page_with_one_call_each_and_one_for_the_pictures() -> None:
    llm = FakeSiteLLM()
    site = await build(llm)
    assert site.pages == ["index", "rooms", "offers", "contact"]
    assert {p: llm.calls[p] for p in site.pages} == dict.fromkeys(site.pages, 1)
    assert llm.calls["images"] == 1
    assert {"index.html", "rooms.html", "assets/base.css", "assets/images.json"} <= set(site.files)
    assert check_site(site.files, WIREFRAME) == []


async def test_the_output_keeps_the_copy_and_the_picture_notes_for_later_edits() -> None:
    site = await build(FakeSiteLLM())
    assert site.copies["rooms"] == PageCopy.model_validate(COPIES["rooms"])
    assert [n.slot for n in site.images] == [s.id for s in SLOTS]
    assert "A calm view for rooms-gallery-2" in site.files["assets/images.json"]


async def test_a_failed_image_call_does_not_fail_the_build() -> None:
    site = await build(FakeSiteLLM(images=WorkerOutputError("no valid notes")))
    assert site.images == []
    assert "Illustration for Real rooms, real light" in site.files["assets/images.json"]


async def test_a_page_that_cannot_be_written_is_named_in_the_error() -> None:
    bad = {**COPIES, "offers": {"sections": []}}  # fails the schema on every try
    with pytest.raises(WorkerOutputError, match="'offers'"):
        await build(FakeSiteLLM(copies=bad))


def test_assemble_stops_and_names_the_files_when_the_site_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = [f"contact.html: problem {n}" for n in range(9)]
    monkeypatch.setattr("webkiln_agents.workers.generate.check_site", lambda *_: found)
    state = GenState(
        task_id="generate",
        instruction="",
        brief=SAMPLE_BRIEF,
        analysis=None,
        brand=SAMPLE_BRAND,
        wireframe=WIREFRAME.model_dump(mode="json"),
        copies=COPIES,
    )
    with pytest.raises(GenerateError, match=r"\(9\).*contact\.html: problem 0.*and 3 more"):
        assemble_node(state)


async def test_the_manager_instruction_reaches_the_page_prompt() -> None:
    llm = FakeSiteLLM()
    await build(llm, instruction="Make the offers page shorter.")
    assert any("<instruction>" in text and "offers page shorter" in text for text in llm.prompts)


async def test_it_needs_the_brand_and_the_wireframe() -> None:
    base = packet("generate")
    base["upstream"] = {"brand": SAMPLE_BRAND}
    with pytest.raises(Exception, match="wireframe"):
        await generate_worker(base, Context(llm=FakeSiteLLM()))


def test_the_registry_has_a_generate_worker() -> None:
    assert WORKERS["generate"] is generate_worker


def test_an_image_plan_for_the_sample_site_is_valid() -> None:
    assert len(ImagePlan.model_validate(IMAGES).images) == len(SLOTS)


async def test_a_crash_in_one_page_reruns_only_that_page_and_assemble() -> None:
    """The subgraph checkpoints with its parent: after a crash, finished nodes do not run again."""
    saver = InMemorySaver()
    config = run_config(Context(), "gen-recovery")
    first = FakeSiteLLM(slow={"offers"})  # `offers` never answers, like a process that dies
    context = Context(llm=first, workers={"generate": generate_worker})
    graph = _single_node_graph(saver)
    run = asyncio.create_task(
        graph.ainvoke(generate_packet(), config, context=context, durability="sync")
    )
    async with asyncio.timeout(5):
        await first.others_done.wait()
    await asyncio.sleep(0.05)  # lets the saver store the finished nodes before the crash
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)

    second = FakeSiteLLM()
    context = Context(llm=second, workers={"generate": generate_worker})
    result = await graph.ainvoke(None, config, context=context, durability="sync")
    assert set(SiteBuild.model_validate(result["data"]).files) >= {"index.html"}
    assert second.calls["offers"] == 1  # the page that crashed
    assert second.calls["index"] == 0  # the finished pages were not written again
    assert second.calls["images"] == 0


def _single_node_graph(saver: InMemorySaver) -> Any:
    from typing import NotRequired

    from langgraph.graph import END, START, StateGraph

    class Holder(
        WorkerInput
    ):  # the packet is the state, so the node reads it as `dispatch` sends it
        data: NotRequired[dict[str, Any]]

    async def node(state: Any, runtime: Any) -> dict:
        result = await generate_worker(state, runtime.context)
        return {"data": result.data}

    builder = StateGraph(Holder, context_schema=Context)  # type: ignore[type-var]
    builder.add_node("run", node)
    builder.add_edge(START, "run")
    builder.add_edge("run", END)
    return builder.compile(checkpointer=saver)


async def test_the_whole_run_builds_the_site_after_the_brand_is_approved() -> None:
    """Real generate worker inside the Manager graph, with fake workers for the others."""
    from webkiln_agents.testing import fake_workers, run_approved

    async def wireframe(_: Any, __: Any) -> Any:
        from webkiln_agents.workers.base import WorkerResult

        return WorkerResult(summary="wireframe", data=WIREFRAME.model_dump(mode="json"))

    async def brand(_: Any, __: Any) -> Any:
        from webkiln_agents.workers.base import WorkerResult

        return WorkerResult(summary="brand", data=SAMPLE_BRAND)

    workers = {
        **fake_workers(),
        "wireframe": wireframe,
        "brand": brand,
        "generate": generate_worker,
    }
    context = fake_context(workers=workers, llm=FakeSiteLLM())
    values = await run_approved(context, SAMPLE_BRIEF, thread_id="whole")
    assert values["tasks"]["generate"].status == "done"
    site = SiteBuild.model_validate(values["outputs"]["generate"])
    assert Wireframe.model_validate(WIREFRAME).pages[0].slug in site.pages
    assert check_site(site.files, WIREFRAME) == []
    assert "Built 4 pages" in values["summaries"]["generate"]


async def test_progress_events_reach_the_parent_stream() -> None:
    graph = _single_node_graph(InMemorySaver())
    context = Context(llm=FakeSiteLLM())
    config = run_config(context, "events")
    events = [
        chunk
        async for chunk in graph.astream(
            generate_packet(), config, context=context, stream_mode="custom"
        )
    ]
    written = {e["page"] for e in events if e["event"] == "page_written"}
    assert written == {"index", "rooms", "offers", "contact"}
    assert {e["event"] for e in events} >= {"images_written", "site_assembled"}
    assert {e["agent"] for e in events} == {"generate"}


def test_the_subgraph_has_the_expected_nodes() -> None:
    assert {"write_page", "write_images", "assemble"} <= set(GENERATE_GRAPH.get_graph().nodes)
