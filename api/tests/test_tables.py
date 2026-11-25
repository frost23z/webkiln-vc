from alembic import command
from sqlalchemy import create_engine, inspect
from webkiln_api.db import alembic_config, sqlalchemy_url


def test_migration_creates_the_three_tables() -> None:
    engine = create_engine(sqlalchemy_url())
    try:
        tables = set(inspect(engine).get_table_names())
        columns = {c["name"] for c in inspect(engine).get_columns("tasks")}
    finally:
        engine.dispose()
    assert {"projects", "runs", "tasks"} <= tables
    assert {"run_id", "id", "agent", "depends_on", "status", "attempts", "error"} <= columns


def test_migration_can_be_undone_and_applied_again() -> None:
    config = alembic_config()
    command.downgrade(config, "base")
    engine = create_engine(sqlalchemy_url())
    try:
        assert "tasks" not in inspect(engine).get_table_names()
    finally:
        engine.dispose()
    command.upgrade(config, "head")
