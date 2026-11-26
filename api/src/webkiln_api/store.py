"""Everything the API reads and writes in its own Postgres tables."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.plan import Task

from webkiln_api.models import RunStatus, TaskReport
from webkiln_api.tables import BrandVersion, Project, Run, TaskOutput, TaskRow

type Sessions = async_sessionmaker[AsyncSession]


@dataclass(frozen=True)
class StoredRun:
    thread_id: str
    status: RunStatus
    brief: Input
    updated_at: datetime
    project_id: str = ""
    user_id: str | None = None


def stored_run(run: Run, project: Project) -> StoredRun:
    brief = Input(brand_name=project.brand_name, prompt=project.prompt, niche=project.niche)
    return StoredRun(
        run.thread_id,
        cast(RunStatus, run.status),
        brief,
        run.updated_at,
        str(project.id),
        project.user_id,
    )


async def create_run_rows(
    sessions: Sessions, thread_id: str, brief: Input, user_id: str | None = None
) -> None:
    """Save the project and its run. This happens before the graph starts, so a run that dies
    early is still there to list, to resume or to retry."""
    async with sessions.begin() as session:
        project = Project(
            brand_name=brief.brand_name, niche=brief.niche, prompt=brief.prompt, user_id=user_id
        )
        session.add(project)
        await session.flush()  # fills project.id, which the run points to
        session.add(Run(project_id=project.id, thread_id=thread_id, status="running"))


async def find_run(sessions: Sessions, thread_id: str) -> StoredRun | None:
    async with sessions() as session:
        row = (
            await session.execute(
                select(Run, Project)
                .join(Project, Run.project_id == Project.id)
                .where(Run.thread_id == thread_id)
            )
        ).first()
    return stored_run(*row) if row else None


async def list_runs(sessions: Sessions, limit: int) -> list[StoredRun]:
    """The runs changed most recently first."""
    async with sessions() as session:
        rows = await session.execute(
            select(Run, Project)
            .join(Project, Run.project_id == Project.id)
            .order_by(Run.updated_at.desc(), Run.id.desc())
            .limit(limit)
        )
        return [stored_run(*row) for row in rows]


async def unfinished_runs(sessions: Sessions) -> list[StoredRun]:
    """The runs a stopped API process left at `running`."""
    async with sessions() as session:
        rows = await session.execute(
            select(Run, Project)
            .join(Project, Run.project_id == Project.id)
            .where(Run.status == "running")
        )
        return [stored_run(*row) for row in rows]


async def set_status(sessions: Sessions, thread_id: str, status: RunStatus) -> None:
    async with sessions.begin() as session:
        await session.execute(update(Run).where(Run.thread_id == thread_id).values(status=status))


async def sync_run(
    sessions: Sessions,
    thread_id: str,
    status: RunStatus,
    tasks: Mapping[str, Task],
    summaries: Mapping[str, str],
    outputs: Mapping[str, dict[str, Any]],
    reviews: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Bring the run's rows in line with the graph state, in one transaction.

    The checkpoint is the truth and these rows are a copy for queries, so the same call works
    after every update, after a resume and at the end. Rows are inserted or updated by key.
    """
    async with sessions.begin() as session:
        run_id = await session.scalar(select(Run.id).where(Run.thread_id == thread_id))
        if run_id is None:
            return
        await session.execute(update(Run).where(Run.id == run_id).values(status=status))
        if tasks:
            insert_tasks = insert(TaskRow).values(
                [
                    {
                        "run_id": run_id,
                        "id": task.id,
                        "agent": task.agent,
                        "instruction": task.instruction,
                        "status": task.status,
                        "depends_on": task.depends_on,
                        "attempts": task.attempts,
                        "error": task.error,
                    }
                    for task in tasks.values()
                ]
            )
            await session.execute(
                insert_tasks.on_conflict_do_update(
                    index_elements=["run_id", "id"],
                    set_={
                        "instruction": insert_tasks.excluded.instruction,
                        "status": insert_tasks.excluded.status,
                        "attempts": insert_tasks.excluded.attempts,
                        "error": insert_tasks.excluded.error,
                    },
                )
            )
        # Only tasks that are in the table can have an output, so `tasks` is written first.
        saved = {task_id: data for task_id, data in outputs.items() if task_id in tasks}
        if saved:
            insert_outputs = insert(TaskOutput).values(
                [
                    {
                        "run_id": run_id,
                        "task_id": task_id,
                        "summary": summaries.get(task_id, ""),
                        "data": data,
                    }
                    for task_id, data in saved.items()
                ]
            )
            await session.execute(
                insert_outputs.on_conflict_do_update(
                    index_elements=["run_id", "task_id"],
                    set_={
                        "summary": insert_outputs.excluded.summary,
                        "data": insert_outputs.excluded.data,
                    },
                )
            )
        if reviews:
            # A decision is never changed, so a repeat of the sync leaves the saved rows alone.
            await session.execute(
                insert(BrandVersion)
                .values(
                    [
                        {
                            "run_id": run_id,
                            "version": entry["version"],
                            "guideline": entry["guideline"],
                            "decision": entry["decision"],
                            "feedback": entry["feedback"],
                            "edits": entry["edits"],
                        }
                        for entry in reviews
                    ]
                )
                .on_conflict_do_nothing(index_elements=["run_id", "version"])
            )


async def claim_resume(sessions: Sessions, thread_id: str, key: str | None) -> bool:
    """Move a run from `waiting` to `running` in one atomic update. Only one caller wins.

    Two Approve requests at once both read `waiting`, but the database lets one UPDATE match the
    row, so the graph is resumed once. The loser gets `False`.
    """
    async with sessions.begin() as session:
        claimed = await session.scalar(
            update(Run)
            .where(Run.thread_id == thread_id, Run.status == "waiting")
            .values(status="running", resume_key=key)
            .returning(Run.id)
        )
    return claimed is not None


async def last_resume_key(sessions: Sessions, thread_id: str) -> str | None:
    async with sessions() as session:
        return await session.scalar(select(Run.resume_key).where(Run.thread_id == thread_id))


async def find_report(sessions: Sessions, thread_id: str, task_id: str) -> TaskReport | None:
    async with sessions() as session:
        row = (
            await session.execute(
                select(TaskRow, TaskOutput)
                .join(Run, Run.id == TaskRow.run_id)
                .outerjoin(
                    TaskOutput,
                    (TaskOutput.run_id == TaskRow.run_id) & (TaskOutput.task_id == TaskRow.id),
                )
                .where(Run.thread_id == thread_id, TaskRow.id == task_id)
            )
        ).first()
    if row is None:
        return None
    task, output = row
    return TaskReport(
        task_id=task.id,
        agent=task.agent,
        status=task.status,
        attempts=task.attempts,
        error=task.error,
        summary=output.summary if output else None,
        output=output.data if output else None,
    )
