from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from webkiln_agents.manager.graph import build_graph

from webkiln_api.config import get_api_settings
from webkiln_api.errors import register_error_handlers
from webkiln_api.models import RunRecord
from webkiln_api.runs import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.runs = dict[str, RunRecord]()
    async with AsyncPostgresSaver.from_conn_string(get_api_settings().DB_URL) as saver:
        await saver.setup()  # creates the checkpoint tables the first time, then does nothing
        app.state.graph = build_graph(checkpointer=saver)  # Compile once at startup
        yield


def create_app() -> FastAPI:
    app = FastAPI(title="Webkiln API", lifespan=lifespan)

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
