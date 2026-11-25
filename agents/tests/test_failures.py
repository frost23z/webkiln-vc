from collections import Counter
from typing import Any, cast

import httpx
import pytest
from langgraph.errors import NodeTimeoutError
from langgraph.types import RetryPolicy
from webkiln_agents.manager.context import Context, run_config
from webkiln_agents.manager.graph import build_graph
from webkiln_agents.testing import SAMPLE_BRIEF, fake_context, fake_worker, fake_workers
from webkiln_agents.workers.base import WorkerInput, WorkerResult
from webkiln_agents.workers.errors import TransientError, is_transient

# The real policy, with pauses short enough for a test.
FAST_RETRY = RetryPolicy(
    max_attempts=3, initial_interval=0.001, jitter=False, retry_on=is_transient
)


class ApiError(Exception):
    """Looks like an SDK error: the status is an attribute."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class GoogleLikeError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"code {code}")
        self.code = code


@pytest.mark.parametrize(
    "error",
    [
        TransientError("rate limit"),
        TimeoutError(),
        ConnectionError(),
        httpx.ConnectTimeout("slow"),
        NodeTimeoutError("run_worker", 300.0, kind="run", run_timeout=300.0),
        ApiError(429),
        ApiError(503),
        GoogleLikeError(429),
    ],
)
def test_transient_errors_are_retried(error: Exception) -> None:
    assert is_transient(error)


@pytest.mark.parametrize(
    "error",
    [ValueError("bad"), KeyError("x"), ApiError(400), ApiError(401), GoogleLikeError(403)],
)
def test_other_errors_are_not_retried(error: Exception) -> None:
    assert not is_transient(error)


async def run(context: Context) -> dict[str, Any]:
    graph = build_graph(worker_retry=FAST_RETRY)
    return await graph.ainvoke(SAMPLE_BRIEF, run_config(context), context=context)


def failing(agent: str, error: Exception, times: int | None = None) -> tuple[Context, Counter[str]]:
    """A context where `agent` raises `error` (the first `times` calls, or always)."""
    calls = Counter[str]()

    async def worker(packet: WorkerInput, context: Context) -> WorkerResult:
        name = packet["task"]["agent"]
        calls[name] += 1
        if name == agent and (times is None or calls[name] <= times):
            raise error
        return await fake_worker(packet, context)

    return fake_context(workers=dict.fromkeys(fake_workers(), worker)), calls


async def test_a_transient_error_is_retried_and_only_that_worker_runs_again() -> None:
    context, calls = failing("seo", TransientError("rate limit"), times=2)

    result = await run(context)

    assert calls["seo"] == 3
    assert calls["niche"] == calls["audience"] == calls["competitor"] == 1
    assert result["tasks"]["seo"].status == "done"


async def test_a_transient_error_that_never_passes_fails_the_task_after_three_tries() -> None:
    context, calls = failing("seo", TransientError("rate limit"))

    result = await run(context)

    assert calls["seo"] == 3
    assert result["tasks"]["seo"].status == "failed"


async def test_a_non_transient_error_is_not_retried() -> None:
    context, calls = failing("seo", ValueError("bad output"))

    result = await run(context)

    assert calls["seo"] == 1
    assert result["tasks"]["seo"].status == "failed"
    assert result["tasks"]["seo"].error == "ValueError: bad output"


async def test_one_failure_leaves_siblings_done_and_dependents_pending() -> None:
    context, calls = failing("competitor", ValueError("boom"))

    result = await run(context)

    statuses = {task.id: task.status for task in result["tasks"].values()}
    assert statuses == {
        "niche": "done",
        "audience": "done",
        "seo": "done",
        "competitor": "failed",
        "brand": "pending",
        "wireframe": "pending",
        "generate": "pending",
    }
    assert "brand" not in calls
    assert set(result["summaries"]) == {"niche", "audience", "seo"}


async def test_the_error_text_is_cut_short() -> None:
    context, _ = failing("seo", ValueError("x" * 1000))

    result = await run(context)

    assert len(result["tasks"]["seo"].error) == 300


async def test_each_worker_reports_its_progress_on_the_custom_stream() -> None:
    context, _ = failing("seo", ValueError("boom"))
    graph = build_graph(worker_retry=FAST_RETRY)

    events = [
        cast(dict[str, Any], chunk["data"])
        async for chunk in graph.astream(
            SAMPLE_BRIEF, run_config(context), context=context, stream_mode="custom", version="v2"
        )
    ]

    by_task = {(e["task_id"], e["event"]) for e in events}
    assert {("niche", "worker_started"), ("niche", "worker_done")} <= by_task
    assert ("seo", "worker_started") in by_task
    assert ("seo", "worker_failed") in by_task
    assert ("seo", "worker_done") not in by_task
    failed = next(e for e in events if e["event"] == "worker_failed")
    assert failed["error"] == "ValueError: boom"
