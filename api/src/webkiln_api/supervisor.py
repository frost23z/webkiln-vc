import asyncio
import logging
from dataclasses import replace
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.context import Context, run_config
from webkiln_agents.manager.editing import EDIT_NODES, edit_command
from webkiln_agents.manager.review import ReviewDecision
from webkiln_agents.manager.state import State

from webkiln_api import store
from webkiln_api.memory import MemoryFactory
from webkiln_api.models import RunStatus
from webkiln_api.site import strip_update
from webkiln_api.status import status_of
from webkiln_api.store import Sessions
from webkiln_api.streaming import RunChannel, sse

logger = logging.getLogger(__name__)

type Graph = CompiledStateGraph[State, Context, Input, State]


def chat_of(messages: list[AnyMessage]) -> list[tuple[str, str]]:
    """The chat as `(role, text)`. Only what the user wrote and what the Manager answered."""
    chat = list[tuple[str, str]]()
    for message in messages:
        if isinstance(message, HumanMessage | AIMessage):
            text = message.content if isinstance(message.content, str) else str(message.content)
            chat.append(("user" if isinstance(message, HumanMessage) else "assistant", text))
    return chat


class RunSupervisor:
    """Runs graphs in the background, so no run depends on a client staying connected.

    Each running run has a `RunChannel` that clients follow. The graph saves a checkpoint after
    every step (`durability="sync"`), so a run that was cut off can be resumed from it.
    """

    def __init__(
        self,
        graph: Graph,
        sessions: Sessions,
        context: Context,
        memory: MemoryFactory | None = None,
    ) -> None:
        self.graph = graph
        self.sessions = sessions
        self.context = context
        self.memory = memory
        self.channels = dict[str, RunChannel]()
        self._tasks = set[asyncio.Task[None]]()

    def start(self, thread_id: str, brief: Input | None, editing: bool = False) -> None:
        """Run in the background. With a brief the run begins, and without one it resumes from
        its checkpoint. `editing` says the run is handling a chat message, so a failure leaves it
        finished instead of failed."""
        self._launch(thread_id, brief.model_dump() if brief else None, editing=editing)

    def resume(self, thread_id: str, decision: ReviewDecision) -> None:
        """Answer the review the run is waiting at. The graph carries on from its interrupt."""
        answer = decision.model_dump(mode="json", exclude_none=True)
        self._launch(thread_id, Command(resume=answer))

    def send_message(self, thread_id: str, text: str, base_version: int | None) -> None:
        """Hand a chat message to the finished thread. The graph enters at `edit_router`."""
        self._launch(thread_id, edit_command(text, base_version), editing=True)

    def _launch(
        self, thread_id: str, graph_input: dict[str, Any] | Command | None, editing: bool = False
    ) -> None:
        channel = self.channels[thread_id] = RunChannel()
        task = asyncio.create_task(self._run(thread_id, graph_input, channel, editing))
        self._tasks.add(task)  # the loop only keeps a weak reference to a task
        task.add_done_callback(self._tasks.discard)

    def is_running(self, thread_id: str) -> bool:
        return thread_id in self.channels

    async def _context_for(self, thread_id: str) -> Context:
        """The shared context with this run's memory: the user and the project it belongs to.

        A run with no user id, or an API with no memory store, runs without memory. A failure
        here is logged and has the same result, because a run never depends on memory.
        """
        if self.memory is None:
            return self.context
        try:
            run = await store.find_run(self.sessions, thread_id)
            if run is not None and run.user_id:
                return replace(self.context, memory=self.memory(run.user_id, run.project_id))
        except Exception:
            logger.exception("Could not set up the memory of run %s", thread_id)
        return replace(self.context, memory=None)

    async def _run(
        self,
        thread_id: str,
        graph_input: dict[str, Any] | Command | None,
        channel: RunChannel,
        editing: bool = False,
    ) -> None:
        config = run_config(self.context, thread_id)
        status: RunStatus = "failed"
        try:
            context = await self._context_for(thread_id)
            async for chunk in self.graph.astream(
                graph_input,
                config,
                context=context,
                stream_mode=["updates", "custom"],
                version="v2",
                durability="sync",
            ):
                data = strip_update(chunk["data"]) if chunk["type"] == "updates" else chunk["data"]
                channel.publish(sse(chunk["type"], data))
                if chunk["type"] == "updates":
                    await self.sync(thread_id, "running")
            snapshot = await self.graph.aget_state(config)
            status = status_of(snapshot)
            if status == "running":  # the graph stopped with steps left, which nothing resumes
                status = "failed"
            await self.sync(thread_id, status)
        except asyncio.CancelledError:
            # The API is shutting down. The status stays `running`, so the next start resumes it.
            raise
        except Exception:
            # The 200 is already sent, so the failure travels as an event. Details stay in the log.
            logger.exception("Run %s failed", thread_id)
            # A message that broke the graph loses nothing: the site is as the last checkpoint
            # saved it. The run stays finished, so the user can ask again, and does not become a
            # failed run that no retry could fix.
            await self._save_status(thread_id, "completed" if editing else "failed")
            channel.publish(sse("error", {"code": "run_failed", "message": "The run failed."}))
        else:
            channel.publish(sse("done", {"status": status}))
        finally:
            channel.finish()
            self.channels.pop(thread_id, None)

    async def sync(self, thread_id: str, status: RunStatus) -> None:
        """Copy the graph state into the run's rows. A failure is logged and does not stop the
        run, because the checkpoint holds everything and the next sync repeats the work."""
        try:
            snapshot = await self.graph.aget_state(run_config(self.context, thread_id))
            values = snapshot.values
            await store.sync_run(
                self.sessions,
                thread_id,
                status,
                values.get("tasks", {}),
                values.get("summaries", {}),
                values.get("outputs", {}),
                values.get("brand_reviews", []),
                values.get("site_changes", []),
                chat_of(values.get("messages", [])),
            )
        except Exception:
            logger.exception("Could not save the rows of run %s", thread_id)

    async def _save_status(self, thread_id: str, status: RunStatus) -> None:
        try:
            await store.set_status(self.sessions, thread_id, status)
        except Exception:
            logger.exception("Could not save the status of run %s", thread_id)

    async def resume_unfinished(self) -> None:
        """Startup sweep: continue every run that a stopped process left at `running`.

        A run with a pending step resumes from its checkpoint with no input and no state reset,
        so finished workers are not repeated. A run that never got a checkpoint starts again from
        its brief. A run whose last step was saved just before the stop only needs its status.
        A run that is `waiting` is not touched: nothing is lost, and the user's answer resumes it.
        A run at `running` that is paused at an interrupt lost its answer in the stop, so it goes
        back to `waiting` and the user answers again.
        """
        for run in await store.unfinished_runs(self.sessions):
            try:
                snapshot = await self.graph.aget_state(run_config(self.context, run.thread_id))
                if not snapshot.values:
                    self.start(run.thread_id, run.brief)
                elif snapshot.interrupts:
                    await self.sync(run.thread_id, "waiting")
                elif snapshot.next:
                    # A step left in an edit node is an edit that the stop cut off.
                    editing = not EDIT_NODES.isdisjoint(snapshot.next)
                    self.start(run.thread_id, None, editing=editing)
                else:
                    await self.sync(run.thread_id, status_of(snapshot))
            except Exception:
                logger.exception("Could not resume run %s", run.thread_id)
                await self._save_status(run.thread_id, "failed")

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
