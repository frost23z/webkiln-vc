from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver
from webkiln_agents.manager.graph import build_graph

from webkiln_api.errors import register_error_handlers
from webkiln_api.models import RunRecord
from webkiln_api.runs import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.graph = build_graph(checkpointer=InMemorySaver())  # Compile once at startup
    app.state.runs = dict[str, RunRecord]()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Webkiln API", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    register_error_handlers(app)
    app.include_router(router)

    return app
