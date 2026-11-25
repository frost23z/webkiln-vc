from typing import Literal

from langgraph.config import get_stream_writer
from langgraph.errors import NodeError
from langgraph.runtime import Runtime
from langgraph.types import Command, Send

from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.analysis import analyze_brief
from webkiln_agents.manager.context import Context
from webkiln_agents.manager.default_plan import default_plan
from webkiln_agents.manager.failures import apply_decision, ask_manager, enforce
from webkiln_agents.manager.plan import Task
from webkiln_agents.manager.scheduler import ready_tasks
from webkiln_agents.manager.state import State
from webkiln_agents.workers.base import WorkerInput

SKIPPED_NOTE = "(skipped: no result)"


def process_initial_input(state: Input) -> dict:
    return {"initial_input": state}


def seed_tasks(state: State) -> dict:
    """Start every run from the default plan and turn it into trackable task state.

    The Manager model does not plan the first pass: the catalog's `after` rules already fix the
    order, so a model call here would only risk a bad plan. The Manager plans when there is
    something to decide, which is feedback on the brand and edits to the finished site.
    """
    plan = default_plan()
    return {"plan": plan, "tasks": {t.id: Task.from_planned(t) for t in plan.tasks}}


async def analyze(state: State, runtime: Runtime[Context]) -> dict:
    """One Manager call that reads the brief for the workers. A failure leaves it empty."""
    llm = runtime.context.llm if runtime.context else None
    analysis = await analyze_brief(llm, state.initial_input)
    return {"analysis": analysis}


def worker_packet(task: Task, state: State) -> WorkerInput:
    """What one worker gets: its task, the brief and the summaries of its dependencies only."""
    context = {
        dep: "(skipped: no result)"
        if state.tasks[dep].status == "skipped"
        else state.summaries[dep]
        for dep in task.depends_on
    }
    return {
        "task": task.model_dump(),
        "brief": state.initial_input.model_dump(),
        "analysis": state.analysis.model_dump() if state.analysis else None,
        "context": context,
        # A task sent back for rework sees its last answer, so it changes only what was asked.
        "previous": state.outputs.get(task.id) if task.status == "needs_rework" else None,
    }


def dispatch(state: State, runtime: Runtime[Context]) -> Command[Literal["run_worker", "__end__"]]:
    """The scheduler. Marks every ready task `running` and starts them in one atomic step.

    The update and the `Send` objects are one result, so a crash cannot leave a task started but
    unmarked. All the `Send` workers run in the same step, and `run_worker -> dispatch` fires once
    per wave, after the last of them. A ready task whose agent has no worker is left `pending`.
    """
    workers = runtime.context.workers if runtime.context else {}
    ready = [task for task in ready_tasks(state.tasks) if task.agent in workers]
    if not ready:
        return Command(goto="__end__")  # END, written as a literal so the return type checks
    running = {
        task.id: task.model_copy(
            update={"status": "running", "attempts": task.attempts + 1, "error": None}
        )
        for task in ready
    }
    sends = [Send("run_worker", worker_packet(task, state)) for task in running.values()]
    return Command(update={"tasks": running}, goto=sends)


def progress(event: str, task: Task, **extra: str) -> None:
    """Send a progress event to the stream. It is not state, so it needs no checkpoint."""
    get_stream_writer()({"event": event, "task_id": task.id, "agent": task.agent, **extra})


async def run_worker(state: WorkerInput, runtime: Runtime[Context]) -> dict:
    """Run one task's worker. Retries and failures are handled around this node (see graph.py)."""
    context = runtime.context or Context()
    task = Task(**state["task"])
    progress("worker_started", task)
    result = await context.workers[task.agent](state, context)
    done = task.model_copy(update={"status": "done", "error": None})
    progress("worker_done", task)
    return {
        "tasks": {task.id: done},
        "summaries": {task.id: result.summary},
        "outputs": {task.id: result.data},
    }


def mark_failed(state: WorkerInput, error: NodeError) -> Command[Literal["review_failures"]]:
    """The error handler of `run_worker`, called once the retries are used up.

    It records the failure and goes on to `review_failures`. Siblings that finished stay `done`,
    and tasks that depend on this one stay `pending`. A handler must never raise, so it keeps to
    plain code. The error text is cut short and holds no traceback.
    """
    task = Task(**state["task"])
    message = f"{type(error.error).__name__}: {error.error}"[:300]
    progress("worker_failed", task, error=message)
    failed = task.model_copy(update={"status": "failed", "error": message})
    return Command(update={"tasks": {task.id: failed}}, goto="review_failures")


async def review_failures(state: State, runtime: Runtime[Context]) -> Command[Literal["dispatch"]]:
    """The Manager's decision on every failed task, then back to `dispatch`.

    Runs once after each wave, whether or not a worker failed, and does nothing when none did. One
    Manager call covers all failures of the wave. A task is retried (`needs_rework`), skipped, or
    handed to the user (`needs_user`). The model proposes and `enforce` applies the rules, so a
    missing, late or broken answer ends as a question to the user, never as a loop.
    """
    failed = sorted((t for t in state.tasks.values() if t.status == "failed"), key=lambda t: t.id)
    if not failed:
        return Command(goto="dispatch")
    llm = runtime.context.llm if runtime.context else None
    proposed = await ask_manager(llm, failed, state.initial_input)
    updates: dict[str, Task] = {}
    for task in failed:
        decision = enforce(task, proposed.get(task.id))
        updates[task.id] = apply_decision(task, decision)
        progress("failure_decision", task, action=decision.action, reason=decision.reason)
    return Command(update={"tasks": updates}, goto="dispatch")
