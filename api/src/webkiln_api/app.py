from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from webkiln_agents.manager.context import Context
from webkiln_agents.manager.graph import build_graph

from webkiln_api.checkpoint import serde
from webkiln_api.config import get_api_settings
from webkiln_api.context import build_context
from webkiln_api.db import create_engine, create_session_factory
from webkiln_api.errors import register_error_handlers
from webkiln_api.runs import router
from webkiln_api.supervisor import RunSupervisor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    async with AsyncPostgresSaver.from_conn_string(get_api_settings().DB_URL, serde=serde) as saver:
        await saver.setup()  # creates the checkpoint tables the first time, then does nothing
        app.state.graph = build_graph(checkpointer=saver)  # Compile once at startup
        engine = create_engine()
        app.state.sessions = create_session_factory(engine)
        app.state.supervisor = RunSupervisor(app.state.graph, app.state.sessions, app.state.context)
        try:
            await app.state.supervisor.resume_unfinished()
            yield
        finally:
            await app.state.supervisor.shutdown()  # runs stay `running` and resume at next start
            await engine.dispose()


def create_app(context: Context | None = None) -> FastAPI:
    """`context` is what runs get to work with. Tests pass fake workers, and the default is the
    real ones."""
    app = FastAPI(title="Webkiln API", lifespan=lifespan)
    app.state.context = context or build_context()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[get_api_settings().UI_ORIGIN],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    register_error_handlers(app)
    app.include_router(router)

    return app
