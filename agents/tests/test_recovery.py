import asyncio
from collections import Counter

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from webkiln_agents.manager.context import Context, run_config
from webkiln_agents.manager.graph import build_graph
from webkiln_agents.testing import APPROVE, SAMPLE_BRIEF, fake_context, fake_worker, fake_workers
from webkiln_agents.workers.base import WorkerInput, WorkerResult

RESEARCH = {"audience", "competitor", "niche", "seo"}


async def test_resume_after_a_crash_runs_only_the_unfinished_workers() -> None:
    saver = InMemorySaver()  # stands in for Postgres: it outlives the first graph run
    calls = Counter[str]()
    finished = set[str]()
    three_done = asyncio.Event()

    async def hangs_on_seo(packet: WorkerInput, context: Context) -> WorkerResult:
        agent = packet["task"]["agent"]
        calls[agent] += 1
        if agent == "seo":
            await asyncio.Event().wait()  # never returns, like a worker caught by a crash
        result = await fake_worker(packet, context)
        finished.add(agent)
        if len(finished) == len(RESEARCH) - 1:
            three_done.set()
        return result

    async def counts(packet: WorkerInput, context: Context) -> WorkerResult:
        calls[packet["task"]["agent"]] += 1
        return await fake_worker(packet, context)

    config = run_config(fake_context(), "recovery")

    # First process: three research workers finish, then the process dies during the fourth.
    first = fake_context(workers=dict.fromkeys(fake_workers(), hangs_on_seo))
    run = asyncio.create_task(
        build_graph(saver).ainvoke(SAMPLE_BRIEF, config, context=first, durability="sync")
    )
    async with asyncio.timeout(5):
        await three_done.wait()
    await asyncio.sleep(0.05)  # lets the saver store the third worker's result before the crash
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)

    # Second process: new graph and context, same checkpoints. Resume with no input.
    second = fake_context(workers=dict.fromkeys(fake_workers(), counts))
    result = await build_graph(saver).ainvoke(None, config, context=second, durability="sync")

    # The brand is written, and the run waits for the user before the wireframe.
    statuses = {task.id: task.status for task in result["tasks"].values()}
    assert statuses == {
        **dict.fromkeys(RESEARCH | {"brand"}, "done"),
        "wireframe": "pending",
        "generate": "pending",
    }
    assert {agent: calls[agent] for agent in RESEARCH} == {
        "audience": 1,
        "competitor": 1,
        "niche": 1,
        "seo": 2,  # started once before the crash, once after
    }
    assert calls["brand"] == 1
    assert calls["wireframe"] == calls["generate"] == 0

    # Approving on the same thread finishes the run without repeating any finished work.
    final = await build_graph(saver).ainvoke(
        Command(resume=APPROVE), config, context=second, durability="sync"
    )
    assert {task.status for task in final["tasks"].values()} == {"done"}
    assert {agent: calls[agent] for agent in RESEARCH} == {
        a: 2 if a == "seo" else 1 for a in RESEARCH
    }
    assert calls["brand"] == calls["wireframe"] == calls["generate"] == 1
