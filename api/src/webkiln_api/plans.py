from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.plan import Task

from webkiln_api.models import RunStatus
from webkiln_api.tables import Project, Run, TaskRow


async def save_plan(
    sessions: async_sessionmaker[AsyncSession],
    thread_id: str,
    brief: Input,
    status: RunStatus,
    tasks: list[Task],
) -> None:
    """Save the project, its run and the planned tasks in one transaction."""
    async with sessions.begin() as session:
        project = Project(brand_name=brief.brand_name, niche=brief.niche, prompt=brief.prompt)
        session.add(project)
        await session.flush()  # fills project.id, which the run points to
        run = Run(project_id=project.id, thread_id=thread_id, status=status)
        session.add(run)
        await session.flush()
        session.add_all(
            TaskRow(
                run_id=run.id,
                id=task.id,
                agent=task.agent,
                instruction=task.instruction,
                status=task.status,
                depends_on=task.depends_on,
                attempts=task.attempts,
                error=task.error,
            )
            for task in tasks
        )


async def save_run_status(
    sessions: async_sessionmaker[AsyncSession], thread_id: str, status: RunStatus
) -> None:
    async with sessions.begin() as session:
        await session.execute(update(Run).where(Run.thread_id == thread_id).values(status=status))
