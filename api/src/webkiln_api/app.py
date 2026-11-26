import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from alembic import command
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from webkiln_agents.manager.context import Context
from webkiln_agents.manager.graph import build_graph

from webkiln_api.checkpoint import serde
from webkiln_api.config import get_api_settings
from webkiln_api.context import build_context
from webkiln_api.db import alembic_config, create_engine, create_session_factory
from webkiln_api.errors import register_error_handlers
from webkiln_api.memory import embeddings_from_settings, index_config, memory_factory
from webkiln_api.runs import router
from webkiln_api.supervisor import RunSupervisor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # Bring the tables to the latest version first, so a fresh database needs no manual step.
    # Alembic runs synchronously, so it gets its own thread. It does nothing when up to date.
    await asyncio.to_thread(command.upgrade, alembic_config(), "head")
    db_url = get_api_settings().DB_URL
    async with AsyncExitStack() as stack:
        saver = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(db_url, serde=serde)
        )
        await saver.setup()  # creates the checkpoint tables the first time, then does nothing
        app.state.graph = build_graph(checkpointer=saver)  # Compile once at startup
        engine = create_engine()
        stack.push_async_callback(engine.dispose)
        app.state.sessions = create_session_factory(engine)
        # Long-term memory needs embeddings: a Google key, or the ones a test passes in. Without
        # them there is no store, and runs work with no memory.
        memory = None
        embeddings = app.state.given_embeddings or embeddings_from_settings()
        if embeddings is not None:
            memory_store = await stack.enter_async_context(
                AsyncPostgresStore.from_conn_string(db_url, index=index_config(embeddings))
            )
            await memory_store.setup()  # creates the store tables and the pgvector index
            memory = memory_factory(memory_store)
        # Tests bring their own context. A real run needs the sessions for the search cache.
        app.state.context = app.state.given_context or build_context(app.state.sessions)
        app.state.supervisor = RunSupervisor(
            app.state.graph, app.state.sessions, app.state.context, memory
        )
        try:
            await app.state.supervisor.resume_unfinished()
            yield
        finally:
            await app.state.supervisor.shutdown()  # runs stay `running` and resume at next start


def create_app(context: Context | None = None, embeddings: Embeddings | None = None) -> FastAPI:
    """`context` is what runs get to work with. Tests pass fake workers, and the default is the
    real ones, built at startup. `embeddings` turns long-term memory on with those embeddings,
    and the default is Google's when a key is set."""
    app = FastAPI(title="Webkiln API", lifespan=lifespan)
    app.state.given_context = context
    app.state.given_embeddings = embeddings

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[get_api_settings().UI_ORIGIN],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Idempotency-Key", "X-User-Id"],
    )
    register_error_handlers(app)
    app.include_router(router)

    return app
