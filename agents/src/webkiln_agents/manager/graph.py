# https://docs.langchain.com/oss/python/langgraph/graph-api-state#define-input-and-output-schemas

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, TimeoutPolicy

from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.approval import brand_review, manager_feedback
from webkiln_agents.manager.context import Context
from webkiln_agents.manager.nodes import (
    analyze,
    dispatch,
    mark_failed,
    process_initial_input,
    review_failures,
    run_worker,
    seed_tasks,
)
from webkiln_agents.manager.remember import remember_approved
from webkiln_agents.manager.state import State
from webkiln_agents.workers.errors import is_transient

# Three tries with a growing pause, for transient errors only (see `is_transient`).
WORKER_RETRY = RetryPolicy(
    max_attempts=3, initial_interval=2.0, max_interval=30.0, retry_on=is_transient
)
# One try may not run longer than this, including the time spent waiting for a rate limit.
WORKER_TIMEOUT = TimeoutPolicy(run_timeout=300)
# Generation writes every page and the pictures' notes, so it gets longer than a research worker.
GENERATE_TIMEOUT = TimeoutPolicy(run_timeout=900)


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    *,
    worker_retry: RetryPolicy = WORKER_RETRY,
) -> CompiledStateGraph[State, Context, Input, State]:
    # Invalid input raises ValidationError which breaks the thread.
    builder = StateGraph(State, context_schema=Context, input_schema=Input)

    builder.add_node("process_initial_input", process_initial_input)
    builder.add_node("seed_tasks", seed_tasks)
    builder.add_node("analyze", analyze)
    builder.add_node("dispatch", dispatch)
    builder.add_node("review_failures", review_failures)
    builder.add_node("brand_review", brand_review)
    builder.add_node("manager_feedback", manager_feedback)
    builder.add_node("remember_approved", remember_approved)
    # LangGraph's type hints have no handler that takes `error: NodeError`, but it injects it.
    # Generation runs the same function and failure handling, with a longer timeout.
    for name, timeout in (("run_worker", WORKER_TIMEOUT), ("run_generate", GENERATE_TIMEOUT)):
        builder.add_node(  # pyrefly: ignore[no-matching-overload]
            name,
            run_worker,
            retry_policy=worker_retry,
            timeout=timeout,
            error_handler=mark_failed,
        )

    builder.add_edge(START, "process_initial_input")
    builder.add_edge("process_initial_input", "seed_tasks")
    builder.add_edge("seed_tasks", "analyze")
    builder.add_edge("analyze", "dispatch")
    # A finished wave goes through `review_failures` (a worker that failed arrives there from
    # `mark_failed`). `dispatch` and `review_failures` choose their next step themselves.
    builder.add_edge("run_worker", "review_failures")
    builder.add_edge("run_generate", "review_failures")
    # Approval is the one way into the second half of the run, and it keeps what was approved.
    builder.add_edge("remember_approved", "dispatch")

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
