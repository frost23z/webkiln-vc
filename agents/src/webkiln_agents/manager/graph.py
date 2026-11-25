# https://docs.langchain.com/oss/python/langgraph/graph-api-state#define-input-and-output-schemas

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.context import Context
from webkiln_agents.manager.nodes import dispatch, process_initial_input, run_worker, seed_tasks
from webkiln_agents.manager.state import State


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph[State, Context, Input, State]:
    # Invalid input raises ValidationError which breaks the thread.
    builder = StateGraph(State, context_schema=Context, input_schema=Input)

    builder.add_node("process_initial_input", process_initial_input)
    builder.add_node("seed_tasks", seed_tasks)
    builder.add_node("dispatch", dispatch)
    builder.add_node("run_worker", run_worker)

    builder.add_edge(START, "process_initial_input")
    builder.add_edge("process_initial_input", "seed_tasks")
    builder.add_edge("seed_tasks", "dispatch")
    # `dispatch` picks its next step itself (`Send` for a wave, or the end), so it has no edge out.
    builder.add_edge("run_worker", "dispatch")

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
