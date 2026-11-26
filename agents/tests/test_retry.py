import asyncio
from collections import Counter
from typing import Any, cast

import pytest
from langgraph.types import RetryPolicy
from webkiln_agents.manager.catalog import CATALOG
from webkiln_agents.manager.context import Context, run_config
from webkiln_agents.manager.graph import build_graph
from webkiln_agents.testing import SAMPLE_BRIEF, fake_context, fake_worker, fake_workers
from webkiln_agents.workers.base import WorkerInput, WorkerResult
from webkiln_agents.workers.errors import TransientError
from webkiln_agents.workers.retry import pause_before, with_retries

LIMIT = TransientError("rate limit")
BAD = ValueError("bad output")
FAST = RetryPolicy(max_attempts=3, initial_interval=0.001, jitter=False)


async def run_with(agent: str, error: Exception, times: int | None = None):
    calls = Counter[str]()
    events = list[dict]()

    async def worker(packet: WorkerInput, context: Context) -> WorkerResult:
        name = packet["task"]["agent"]
        calls[name] += 1
        if name == agent and (times is None or calls[name] <= times):
            raise error
        return await fake_worker(packet, context)

    context = fake_context(workers=dict.fromkeys(fake_workers(), worker))
    graph = build_graph(worker_retry=FAST)
    async for mode, chunk in graph.astream(
        SAMPLE_BRIEF, run_config(context), context=context, stream_mode=["custom", "values"]
    ):
        if mode == "custom":
            events.append(cast(dict[str, Any], chunk))
    return calls, events


def test_every_agent_has_sane_limits() -> None:
    for spec in CATALOG.values():
        assert 1 <= spec.max_tries <= 3, spec.name
        assert spec.attempt_timeout > 0, spec.name
    assert CATALOG["generate"].max_tries < CATALOG["seo"].max_tries  # a new try repeats every page


async def test_a_research_agent_gets_three_tries_for_a_transient_error() -> None:
    calls, _ = await run_with("seo", TransientError("rate limit"))
    assert calls["seo"] == CATALOG["seo"].max_tries == 3


async def test_the_generate_agent_gets_only_two_tries() -> None:
    # Generate needs an approved run to start, so the loop is tested with the agent's own limit.
    attempts = 0

    async def attempt() -> str:
        nonlocal attempts
        attempts += 1
        raise LIMIT

    with pytest.raises(TransientError):
        await with_retries(
            attempt, policy=FAST, max_tries=CATALOG["generate"].max_tries, seconds_per_try=1
        )
    assert attempts == 2


async def test_the_policy_is_a_ceiling_for_every_agent() -> None:
    attempts = 0

    async def attempt() -> None:
        nonlocal attempts
        attempts += 1
        raise LIMIT

    with pytest.raises(TransientError):
        await with_retries(attempt, policy=FAST, max_tries=10, seconds_per_try=1)
    assert attempts == FAST.max_attempts


async def test_a_retry_is_reported_on_the_stream_with_the_try_number() -> None:
    calls, events = await run_with("seo", TransientError("rate limit"), times=1)
    retries = [e for e in events if e.get("event") == "worker_retry"]
    assert calls["seo"] == 2
    assert [(e["agent"], e["tries"]) for e in retries] == [("seo", "1")]


async def test_a_try_that_runs_out_of_time_is_retried_like_any_transient_error() -> None:
    attempts = 0

    async def attempt() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(5)
        return "ok"

    assert await with_retries(attempt, policy=FAST, max_tries=3, seconds_per_try=0.02) == "ok"
    assert attempts == 2


async def test_a_cancel_is_not_swallowed() -> None:
    started = asyncio.Event()

    async def attempt() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(with_retries(attempt, policy=FAST, max_tries=3, seconds_per_try=10))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_non_transient_error_is_raised_on_the_first_try() -> None:
    attempts = 0

    async def attempt() -> None:
        nonlocal attempts
        attempts += 1
        raise BAD

    with pytest.raises(ValueError, match="bad output"):
        await with_retries(attempt, policy=FAST, max_tries=3, seconds_per_try=1)
    assert attempts == 1


def test_the_pause_grows_and_stops_at_the_maximum() -> None:
    policy = RetryPolicy(initial_interval=2.0, backoff_factor=2.0, max_interval=5.0, jitter=False)
    assert [pause_before(policy, n) for n in (1, 2, 3)] == [2.0, 4.0, 5.0]
