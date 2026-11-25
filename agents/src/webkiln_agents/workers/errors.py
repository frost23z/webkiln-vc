import httpx
from langgraph.errors import NodeTimeoutError

# HTTP statuses that mean "try again later" rather than "this request is wrong".
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class TransientError(Exception):
    """A failure that may pass by itself: a rate limit, a timeout, a network error."""


class WorkerOutputError(Exception):
    """The model never produced a valid output, even after the worker's own retries."""


def _status_of(error: Exception) -> int | None:
    """The HTTP status an SDK error carries, as `status_code`, `code` or on its `response`."""
    for holder in (error, getattr(error, "response", None)):
        for name in ("status_code", "code"):
            value = getattr(holder, name, None)
            if isinstance(value, int):
                return value
    return None


def is_transient(error: Exception) -> bool:
    """Whether a worker error is worth retrying. This is the graph's explicit `retry_on`.

    LangGraph's default would retry almost any error, including a 400 or 401 from a model SDK,
    which can never succeed. Here only rate limits, timeouts, server errors and network errors
    count. A validation error is not retried by the graph: the worker retries it itself.
    """
    if isinstance(
        error,
        (TransientError, NodeTimeoutError, TimeoutError, ConnectionError, httpx.TransportError),
    ):
        return True
    return _status_of(error) in RETRYABLE_STATUS
