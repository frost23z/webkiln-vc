"""Retry a worker's transient failures, with the limits of its own agent.

The graph's `RetryPolicy` has one limit for a whole node, so it cannot give each agent its own
(`AgentSpec.max_tries` and `AgentSpec.attempt_timeout`). The loop is here instead, in the worker
node. The policy still sets the pauses and the highest number of tries any agent may have.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable

from langgraph.types import RetryPolicy

from webkiln_agents.workers.errors import is_transient

# Three tries at most, with a growing pause, for transient errors only (see `is_transient`).
WORKER_RETRY = RetryPolicy(
    max_attempts=3, initial_interval=2.0, max_interval=30.0, retry_on=is_transient
)


def pause_before(policy: RetryPolicy, retry: int) -> float:
    """Seconds to wait before retry number `retry` (1 for the first retry)."""
    pause = min(policy.initial_interval * policy.backoff_factor ** (retry - 1), policy.max_interval)
    if not policy.jitter:
        return pause
    return pause + random.uniform(0, 1)  # noqa: S311  (jitter, not security)


async def with_retries[T](
    attempt: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    max_tries: int,
    seconds_per_try: float,
    on_retry: Callable[[int, Exception], None] = lambda _tries, _error: None,
) -> T:
    """Run `attempt` until it succeeds or the tries are used up, then raise its last error.

    Only errors that `is_transient` accepts are retried. Each try has its own time limit
    (`seconds_per_try`), and a try that runs out of time counts as a transient error.
    `max_tries` is cut down to the policy's own limit.
    """
    tries = max(1, min(max_tries, policy.max_attempts))
    number = 0
    while True:
        number += 1
        try:
            async with asyncio.timeout(seconds_per_try):
                return await attempt()
        except Exception as error:
            if number == tries or not is_transient(error):
                raise
            on_retry(number, error)
            await asyncio.sleep(pause_before(policy, number))
