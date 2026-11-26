"""Everything the API reads and writes in its own Postgres tables."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from webkiln_agents.budget import DEFAULT_CALL_LIMIT
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.plan import Task

from webkiln_api.models import RunStatus, TaskReport
from webkiln_api.site import GENERATE_AGENT, INITIAL_CAUSE, RESTORE_CAUSE, without_files
from webkiln_api.tables import (
    BrandVersion,
    ChatMessage,
    Project,
    Run,
    SiteFile,
    SiteVersion,
    TaskOutput,
    TaskRow,
)

type Sessions = async_sessionmaker[AsyncSession]


@dataclass(frozen=True)
class StoredRun:
    thread_id: str
    status: RunStatus
    brief: Input
    updated_at: datetime
    project_id: str = ""
    user_id: str | None = None
    model_calls: int = 0
    call_limit: int = DEFAULT_CALL_LIMIT


def stored_run(run: Run, project: Project) -> StoredRun:
    brief = Input(brand_name=project.brand_name, prompt=project.prompt, niche=project.niche)
    return StoredRun(
        run.thread_id,
        cast(RunStatus, run.status),
        brief,
        run.updated_at,
        str(project.id),
        project.user_id,
        run.model_calls,
        run.call_limit,
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


async def save_calls(sessions: Sessions, thread_id: str, used: int) -> None:
    """Keep the run's count of model calls. It never goes down, so a late write from a stale
    process cannot give calls back."""
    async with sessions.begin() as session:
        await session.execute(
            update(Run)
            .where(Run.thread_id == thread_id)
            .values(model_calls=func.greatest(Run.model_calls, used))
        )


async def allow_calls(sessions: Sessions, thread_id: str, extra: int) -> int | None:
    """Raise the run's call limit by `extra`, and return the new limit (`None`: no such run)."""
    async with sessions.begin() as session:
        return await session.scalar(
            update(Run)
            .where(Run.thread_id == thread_id)
            .values(call_limit=Run.call_limit + extra)
            .returning(Run.call_limit)
        )


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
    site_changes: Sequence[Mapping[str, Any]] = (),
    chat: Sequence[tuple[str, str]] = (),
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
                        "data": without_files(data),
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
        for task_id, data in saved.items():
            if tasks[task_id].agent == GENERATE_AGENT and tasks[task_id].status == "done":
                await _save_first_site(
                    session,
                    run_id,
                    data,
                    outputs.get("wireframe"),
                    outputs.get("brand"),
                    summaries.get(task_id, ""),
                )
        if site_changes:
            await _save_site_changes(session, run_id, site_changes)
        if chat:
            await _save_chat(session, run_id, chat)
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


async def _save_first_site(
    session: AsyncSession,
    run_id: Any,
    build: Mapping[str, Any],
    wireframe: Mapping[str, Any] | None,
    brand: Mapping[str, Any] | None,
    summary: str,
) -> None:
    """Save what `generate` built as site version 1, with all its files.

    The same build is saved again as the same version, so a repeat of the sync changes nothing.
    A build that was reworked replaces version 1, because no edit has been made to it yet.
    """
    if not build.get("files") or wireframe is None:
        return
    row = {
        "run_id": run_id,
        "version": 1,
        "parent_version": None,
        "cause": INITIAL_CAUSE,
        "summary": summary,
        "brand": brand,
        "wireframe": wireframe,
        "copies": build["copies"],
        "images": build["images"],
    }
    insert_version = insert(SiteVersion).values(row)
    await session.execute(
        insert_version.on_conflict_do_update(
            index_elements=["run_id", "version"],
            set_={
                "summary": insert_version.excluded.summary,
                "brand": insert_version.excluded.brand,
                "wireframe": insert_version.excluded.wireframe,
                "copies": insert_version.excluded.copies,
                "images": insert_version.excluded.images,
            },
        )
    )
    await session.execute(delete(SiteFile).where(SiteFile.run_id == run_id, SiteFile.version == 1))
    await session.execute(
        insert(SiteFile).values(
            [
                {"run_id": run_id, "version": 1, "path": path, "content": content}
                for path, content in build["files"].items()
            ]
        )
    )


async def _save_site_changes(
    session: AsyncSession, run_id: Any, changes: Sequence[Mapping[str, Any]]
) -> None:
    """Save the versions that edits made: their inputs, and only the files that changed.

    A saved version is never changed, so a repeat of the sync (or a version saved by an earlier
    sync) leaves the rows alone. The files go in only with a version that was just inserted.
    """
    for change in changes:
        inserted = await session.scalar(
            insert(SiteVersion)
            .values(
                {
                    "run_id": run_id,
                    "version": change["version"],
                    "parent_version": change["parent_version"],
                    "cause": change["cause"],
                    "summary": change["summary"],
                    "brand": change["brand"],
                    "wireframe": change["wireframe"],
                    "copies": change["copies"],
                    "images": change["images"],
                }
            )
            .on_conflict_do_nothing(index_elements=["run_id", "version"])
            .returning(SiteVersion.version)
        )
        if inserted is not None and change["changed"]:
            await session.execute(
                insert(SiteFile).values(
                    [
                        {
                            "run_id": run_id,
                            "version": change["version"],
                            "path": path,
                            "content": content,
                        }
                        for path, content in change["changed"].items()
                    ]
                )
            )


async def _save_chat(session: AsyncSession, run_id: Any, chat: Sequence[tuple[str, str]]) -> None:
    """Copy the chat, keyed by its position in the graph's messages. A message never changes."""
    await session.execute(
        insert(ChatMessage)
        .values(
            [
                {"run_id": run_id, "position": position, "role": role, "content": content}
                for position, (role, content) in enumerate(chat)
            ]
        )
        .on_conflict_do_nothing(index_elements=["run_id", "position"])
    )


@dataclass(frozen=True)
class StoredVersion:
    version: int
    parent_version: int | None
    cause: str
    summary: str
    created_at: datetime


async def list_versions(sessions: Sessions, thread_id: str) -> list[StoredVersion]:
    """The saved versions of a site, newest first."""
    async with sessions() as session:
        rows = await session.execute(
            select(SiteVersion)
            .join(Run, Run.id == SiteVersion.run_id)
            .where(Run.thread_id == thread_id)
            .order_by(SiteVersion.version.desc())
        )
        return [
            StoredVersion(v.version, v.parent_version, v.cause, v.summary, v.created_at)
            for v in rows.scalars()
        ]


async def latest_version(sessions: Sessions, thread_id: str) -> int | None:
    async with sessions() as session:
        return await session.scalar(
            select(func.max(SiteVersion.version))
            .join(Run, Run.id == SiteVersion.run_id)
            .where(Run.thread_id == thread_id)
        )


async def site_files(
    sessions: Sessions, thread_id: str, version: int | None = None
) -> dict[str, str]:
    """The files of a saved site version, by path. `None` is the newest version.

    A version holds only the files it changed, so the files come from walking back through its
    parents and letting each newer version replace the older one. A version that does not exist
    has no files.
    """
    async with sessions() as session:
        run_id = await session.scalar(select(Run.id).where(Run.thread_id == thread_id))
        if run_id is None:
            return {}
        rows = await session.execute(
            select(SiteVersion.version, SiteVersion.parent_version, SiteVersion.cause).where(
                SiteVersion.run_id == run_id
            )
        )
        found = {row.version: (row.parent_version, row.cause) for row in rows}
        current = max(found, default=None) if version is None else version
        chain: list[int] = []  # newest first
        while current is not None and current in found:
            chain.append(current)
            parent, cause = found[current]
            # A restored version holds every file, so nothing before it counts: a file that
            # a later version added and the restore dropped must not come back.
            current = None if cause == RESTORE_CAUSE else parent
        if not chain:
            return {}
        rows = await session.execute(
            select(SiteFile.version, SiteFile.path, SiteFile.content).where(
                SiteFile.run_id == run_id, SiteFile.version.in_(chain)
            )
        )
        by_version = dict[int, dict[str, str]]()
        for row in rows:
            by_version.setdefault(row.version, {})[row.path] = row.content
    files = dict[str, str]()
    for number in reversed(chain):  # oldest first, so a newer file replaces an older one
        files.update(by_version.get(number, {}))
    return dict(sorted(files.items()))


async def site_version_inputs(
    sessions: Sessions, thread_id: str, version: int
) -> dict[str, Any] | None:
    """What a version was rendered from (brand, wireframe, copy and picture notes)."""
    async with sessions() as session:
        row = (
            await session.execute(
                select(SiteVersion)
                .join(Run, Run.id == SiteVersion.run_id)
                .where(Run.thread_id == thread_id, SiteVersion.version == version)
            )
        ).scalar_one_or_none()
    if row is None:
        return None
    return {
        "brand": row.brand,
        "wireframe": row.wireframe,
        "copies": row.copies,
        "images": row.images,
        "summary": row.summary,
    }


async def chat_history(sessions: Sessions, thread_id: str) -> list[tuple[str, str]]:
    """The saved chat, oldest first, as `(role, text)`."""
    async with sessions() as session:
        rows = await session.execute(
            select(ChatMessage.role, ChatMessage.content)
            .join(Run, Run.id == ChatMessage.run_id)
            .where(Run.thread_id == thread_id)
            .order_by(ChatMessage.position)
        )
        return [(row.role, row.content) for row in rows]


async def claim_edit(sessions: Sessions, thread_id: str) -> bool:
    """The per-thread run lock: move a run from `completed` to `running` in one atomic update.

    Of two chat messages at once, the database lets one UPDATE match the row, so the graph
    handles one edit at a time. The loser gets `False` and is told the run is busy.
    """
    async with sessions.begin() as session:
        claimed = await session.scalar(
            update(Run)
            .where(Run.thread_id == thread_id, Run.status == "completed")
            .values(status="running")
            .returning(Run.id)
        )
    return claimed is not None


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
