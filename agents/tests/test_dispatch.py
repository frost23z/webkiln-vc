import asyncio
from collections import defaultdict
from typing import Any, cast

from webkiln_agents.manager.context import Context, run_config
from webkiln_agents.manager.graph import build_graph
from webkiln_agents.testing import SAMPLE_BRIEF, fake_context, fake_worker, fake_workers
from webkiln_agents.workers.base import WorkerInput, WorkerResult

RESEARCH = ["audience", "competitor", "niche", "seo"]


async def run(context: Context) -> dict[str, Any]:
    return await build_graph().ainvoke(SAMPLE_BRIEF, run_config(context), context=context)


async def steps(context: Context) -> dict[int, list[str]]:
    """The nodes that ran in each step of a run."""
    by_step = defaultdict[int, list[str]](list)
    async for chunk in build_graph().astream(
        SAMPLE_BRIEF, run_config(context), context=context, stream_mode="debug", version="v2"
    ):
        data = cast(dict[str, Any], chunk["data"])
        if data["type"] == "task":
            by_step[data["step"]].append(data["payload"]["name"])
    return dict(by_step)


async def test_every_task_ends_done_with_a_summary_and_an_output() -> None:
    result = await run(fake_context())

    assert {task.status for task in result["tasks"].values()} == {"done"}
    assert result["summaries"]["niche"] == "niche summary"
    assert result["outputs"]["niche"] == {"agent": "niche"}
    assert set(result["summaries"]) == set(result["tasks"])


async def test_the_four_research_tasks_run_in_one_step_followed_by_one_dispatch() -> None:
    by_step = await steps(fake_context())

    first_wave = next(nodes for nodes in by_step.values() if "run_worker" in nodes)
    assert first_wave == ["run_worker"] * 4
    wave_step = next(step for step, nodes in by_step.items() if nodes == first_wave)
    # One review of the wave's failures (a no-op here), then one dispatch.
    assert by_step[wave_step + 1] == ["review_failures"]
    assert by_step[wave_step + 2] == ["dispatch"]


async def test_the_research_workers_really_overlap_in_time() -> None:
    barrier = asyncio.Barrier(4)

    async def meets_the_others(packet: WorkerInput, context: Context) -> WorkerResult:
        if packet["task"]["agent"] in RESEARCH:
            # Only returns when all four research workers are waiting here at once.
            await asyncio.wait_for(barrier.wait(), timeout=2)
        return await fake_worker(packet, context)

    context = fake_context(workers=dict.fromkeys(fake_workers(), meets_the_others))

    result = await run(context)

    assert result["tasks"]["seo"].status == "done"


async def test_a_task_starts_only_after_its_dependencies_are_done() -> None:
    finished: list[str] = []

    async def records_its_order(packet: WorkerInput, context: Context) -> WorkerResult:
        agent = packet["task"]["agent"]
        if agent == "brand":
            assert sorted(finished) == RESEARCH
        result = await fake_worker(packet, context)
        finished.append(agent)
        return result

    context = fake_context(workers=dict.fromkeys(fake_workers(), records_its_order))

    await run(context)

    assert finished[-3:] == ["brand", "wireframe", "generate"]


async def test_a_worker_gets_only_the_summaries_of_its_dependencies() -> None:
    packets: dict[str, WorkerInput] = {}

    async def keeps_the_packet(packet: WorkerInput, context: Context) -> WorkerResult:
        packets[packet["task"]["agent"]] = packet
        return await fake_worker(packet, context)

    await run(fake_context(workers=dict.fromkeys(fake_workers(), keeps_the_packet)))

    assert packets["niche"]["context"] == {}
    assert packets["niche"]["brief"] == SAMPLE_BRIEF
    assert packets["brand"]["context"] == {name: f"{name} summary" for name in RESEARCH}
    assert packets["wireframe"]["context"] == {"brand": "brand summary"}


async def test_the_concurrency_cap_holds() -> None:
    running = peak = 0

    async def counts_the_overlap(packet: WorkerInput, context: Context) -> WorkerResult:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1
        return await fake_worker(packet, context)

    workers = dict.fromkeys(fake_workers(), counts_the_overlap)

    await run(fake_context(workers=workers, max_concurrency=2))

    assert peak == 2


async def test_an_agent_without_a_worker_is_left_pending() -> None:
    workers = dict.fromkeys(RESEARCH, fake_worker)  # brand and later have no worker yet

    result = await run(fake_context(workers=workers))

    statuses = {task.id: task.status for task in result["tasks"].values()}
    assert statuses == {
        **dict.fromkeys(RESEARCH, "done"),
        "brand": "pending",
        "wireframe": "pending",
        "generate": "pending",
    }


async def test_a_run_without_a_context_only_plans() -> None:
    result = await build_graph().ainvoke(SAMPLE_BRIEF)

    assert {task.status for task in result["tasks"].values()} == {"pending"}
