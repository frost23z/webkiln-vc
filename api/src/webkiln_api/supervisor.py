import asyncio
import logging

from langgraph.graph.state import CompiledStateGraph
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.context import Context, run_config
from webkiln_agents.manager.state import State

from webkiln_api import store
from webkiln_api.models import RunStatus
from webkiln_api.status import status_of
from webkiln_api.store import Sessions
from webkiln_api.streaming import RunChannel, sse

logger = logging.getLogger(__name__)

type Graph = CompiledStateGraph[State, Context, Input, State]


class RunSupervisor:
    """Runs graphs in the background, so no run depends on a client staying connected.

    Each running run has a `RunChannel` that clients follow. The graph saves a checkpoint after
    every step (`durability="sync"`), so a run that was cut off can be resumed from it.
    """

    def __init__(self, graph: Graph, sessions: Sessions, context: Context) -> None:
        self.graph = graph
        self.sessions = sessions
        self.context = context
        self.channels = dict[str, RunChannel]()
        self._tasks = set[asyncio.Task[None]]()

    def start(self, thread_id: str, brief: Input | None) -> None:
        """Run in the background. With a brief the run begins, and without one it resumes from
        its checkpoint."""
        channel = self.channels[thread_id] = RunChannel()
        task = asyncio.create_task(self._run(thread_id, brief, channel))
        self._tasks.add(task)  # the loop only keeps a weak reference to a task
        task.add_done_callback(self._tasks.discard)

    def is_running(self, thread_id: str) -> bool:
        return thread_id in self.channels

    async def _run(self, thread_id: str, brief: Input | None, channel: RunChannel) -> None:
        config = run_config(self.context, thread_id)
        status: RunStatus = "failed"
        try:
            async for chunk in self.graph.astream(
                brief.model_dump() if brief else None,
                config,
                context=self.context,
                stream_mode=["updates", "custom"],
                version="v2",
                durability="sync",
            ):
                channel.publish(sse(chunk["type"], chunk["data"]))
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
            await self._save_status(thread_id, "failed")
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
        """
        for run in await store.unfinished_runs(self.sessions):
            try:
                snapshot = await self.graph.aget_state(run_config(self.context, run.thread_id))
                if not snapshot.values:
                    self.start(run.thread_id, run.brief)
                elif snapshot.next:
                    self.start(run.thread_id, None)
                else:
                    await self.sync(run.thread_id, status_of(snapshot))
            except Exception:
                logger.exception("Could not resume run %s", run.thread_id)
                await self._save_status(run.thread_id, "failed")

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
