# https://docs.langchain.com/oss/python/langgraph/graph-api-state#define-input-and-output-schemas

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.nodes import process_initial_input, seed_tasks
from webkiln_agents.manager.state import State


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph[State, None, Input, State]:
    # Invalid input raises ValidationError which breaks the thread.
    builder = StateGraph(State, input_schema=Input)

    builder.add_node("process_initial_input", process_initial_input)

    builder.add_node("seed_tasks", seed_tasks)

    builder.add_edge(START, "process_initial_input")
    builder.add_edge("process_initial_input", "seed_tasks")
    builder.add_edge("seed_tasks", END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
