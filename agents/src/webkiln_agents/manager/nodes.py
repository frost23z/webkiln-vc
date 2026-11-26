from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langgraph.config import get_stream_writer
from langgraph.errors import NodeError
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, Send

from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.analysis import analyze_brief
from webkiln_agents.manager.approval import awaiting_brand_review
from webkiln_agents.manager.catalog import CATALOG
from webkiln_agents.manager.context import Context
from webkiln_agents.manager.default_plan import default_plan
from webkiln_agents.manager.failures import apply_decision, ask_manager, enforce
from webkiln_agents.manager.plan import Task
from webkiln_agents.manager.scheduler import ready_tasks
from webkiln_agents.manager.state import State
from webkiln_agents.workers.base import WorkerInput
from webkiln_agents.workers.retry import WORKER_RETRY, with_retries

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


def upstream_outputs(task: Task, state: State) -> dict[str, dict[str, Any]]:
    """The full outputs of the agents this task's agent reads, for the ones that finished."""
    done = {t.agent: t.id for t in state.tasks.values() if t.status == "done"}
    return {
        agent: state.outputs[done[agent]]
        for agent in CATALOG[task.agent].reads
        if agent in done and done[agent] in state.outputs
    }


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
        "upstream": upstream_outputs(task, state),
    }


def dispatch(
    state: State, runtime: Runtime[Context]
) -> Command[Literal["run_worker", "run_generate", "brand_review", "remember_site", "__end__"]]:
    """The scheduler. Marks every ready task `running` and starts them in one atomic step.

    The update and the `Send` objects are one result, so a crash cannot leave a task started but
    unmarked. All the `Send` workers run in the same step, and `run_worker -> dispatch` fires once
    per wave, after the last of them. A ready task whose agent has no worker is left `pending`.
    """
    workers = runtime.context.workers if runtime.context else {}
    ready = [
        task
        for task in ready_tasks(state.tasks)
        if task.agent in workers
        and (state.brand_approved or not CATALOG[task.agent].needs_approval)
    ]
    if not ready:
        if awaiting_brand_review(state):
            return Command(goto="brand_review")
        if any(t.agent == "generate" and t.status == "done" for t in state.tasks.values()):
            return Command(goto="remember_site")  # the site is built: keep its architecture
        return Command(goto="__end__")  # END, written as a literal so the return type checks
    running = {
        task.id: task.model_copy(
            update={"status": "running", "attempts": task.attempts + 1, "error": None}
        )
        for task in ready
    }
    # The packet is built from the task as it was before it started, so a task sent back for
    # rework still shows as `needs_rework` and gets its last answer. The worker sees `running`.
    sends = [
        Send(
            "run_generate" if task.agent == "generate" else "run_worker",
            {**worker_packet(task, state), "task": running[task.id].model_dump()},
        )
        for task in ready
    ]
    return Command(update={"tasks": running}, goto=sends)


def progress(event: str, task: Task, **extra: str) -> None:
    """Send a progress event to the stream. It is not state, so it needs no checkpoint."""
    get_stream_writer()({"event": event, "task_id": task.id, "agent": task.agent, **extra})


def make_run_worker(
    policy: RetryPolicy = WORKER_RETRY,
) -> Callable[[WorkerInput, Runtime[Context]], Awaitable[dict]]:
    """The `run_worker` node, retrying transient errors with each agent's own limits."""

    async def run_worker(state: WorkerInput, runtime: Runtime[Context]) -> dict:
        """Run one task's worker. A failure after the retries goes to `mark_failed`."""
        context = runtime.context or Context()
        task = Task(**state["task"])
        spec = CATALOG[task.agent]
        progress("worker_started", task)

        def retrying(tries: int, error: Exception) -> None:
            progress("worker_retry", task, tries=str(tries), error=f"{error}"[:200])

        result = await with_retries(
            lambda: context.workers[task.agent](state, context),
            policy=policy,
            max_tries=spec.max_tries,
            seconds_per_try=spec.attempt_timeout,
            on_retry=retrying,
        )
        done = task.model_copy(update={"status": "done", "error": None})
        progress("worker_done", task)
        return {
            "tasks": {task.id: done},
            "summaries": {task.id: result.summary},
            "outputs": {task.id: result.data},
        }

    return run_worker


run_worker = make_run_worker()


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
