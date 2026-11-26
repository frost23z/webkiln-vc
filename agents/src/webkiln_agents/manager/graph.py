# https://docs.langchain.com/oss/python/langgraph/graph-api-state#define-input-and-output-schemas

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, TimeoutPolicy

from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.approval import brand_review, manager_feedback
from webkiln_agents.manager.context import Context
from webkiln_agents.manager.editing import (
    edit_content,
    edit_design,
    edit_failed,
    edit_router,
    edit_structure,
    remember_edit,
)
from webkiln_agents.manager.nodes import (
    analyze,
    dispatch,
    make_run_worker,
    mark_failed,
    process_initial_input,
    review_failures,
    seed_tasks,
)
from webkiln_agents.manager.remember import remember_approved, remember_site
from webkiln_agents.manager.state import State
from webkiln_agents.workers.retry import WORKER_RETRY

# Chat edits make one call and cannot be slow, so the graph retries and times them as one node.
WORKER_TIMEOUT = TimeoutPolicy(run_timeout=300)


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
    builder.add_node("remember_site", remember_site)
    # LangGraph's type hints have no handler that takes `error: NodeError`, but it injects it.
    # Both nodes run the same function. It retries and times each try by the agent's own limits
    # (see `AgentSpec`), so the nodes have no graph-level retry policy or timeout.
    for name in ("run_worker", "run_generate"):
        builder.add_node(  # pyrefly: ignore[no-matching-overload]
            name, make_run_worker(worker_retry), error_handler=mark_failed
        )

    # Chat edits enter at `edit_router` with `Command(goto=...)` on the finished thread. There is
    # no branch for them at START: the input schema is the brief, and a first run always has one.
    builder.add_node("edit_router", edit_router)
    builder.add_node("remember_edit", remember_edit)
    for name, edit in (
        ("edit_content", edit_content),
        ("edit_design", edit_design),
        ("edit_structure", edit_structure),
    ):
        builder.add_node(  # pyrefly: ignore[no-matching-overload]
            name,
            edit,
            retry_policy=worker_retry,
            timeout=WORKER_TIMEOUT,
            error_handler=edit_failed,
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
    builder.add_edge("remember_site", END)
    builder.add_edge("remember_edit", END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
