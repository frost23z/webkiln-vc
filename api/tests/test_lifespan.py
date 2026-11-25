from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from webkiln_api.app import create_app
from webkiln_api.db import alembic_config, sqlalchemy_url


def test_graph_is_compiled_once_at_startup() -> None:
    app = create_app()

    with TestClient(app) as client:
        graph = app.state.graph
        client.get("/health")
        client.get("/health")

        assert graph is not None
        assert app.state.graph is graph  # not rebuilt per request


def test_startup_creates_the_tables_of_an_empty_database() -> None:
    config = alembic_config()
    command.downgrade(config, "base")
    engine = create_engine(sqlalchemy_url())
    try:
        assert "runs" not in inspect(engine).get_table_names()
        with TestClient(create_app()) as client:
            assert client.get("/health").status_code == 200
        assert {"projects", "runs", "tasks", "task_outputs", "search_cache"} <= set(
            inspect(engine).get_table_names()
        )
    finally:
        engine.dispose()
