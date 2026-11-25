import os
from collections.abc import Iterator
from urllib.parse import urlparse

import psycopg
import pytest
from alembic import command
from psycopg import sql
from psycopg.conninfo import make_conninfo
from webkiln_api.config import get_api_settings
from webkiln_api.db import alembic_config

# A separate database, so tests never touch the data of the dev database.
TEST_DB_URL = os.environ.get(
    "TEST_DB_URL", "postgresql://webkiln:webkiln@localhost:5432/webkiln_test"
)


@pytest.fixture(scope="session", autouse=True)
def _test_database() -> None:
    """Create the test database on the compose server if it does not exist yet."""
    name = urlparse(TEST_DB_URL).path.lstrip("/")
    assert name.endswith("_test"), "TEST_DB_URL must name a database that ends in _test"
    with psycopg.connect(make_conninfo(TEST_DB_URL, dbname="postgres"), autocommit=True) as conn:
        found = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if found is None:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


@pytest.fixture(autouse=True)
def _use_test_database(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DB_URL", TEST_DB_URL)
    get_api_settings.cache_clear()
    yield
    get_api_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def _migrated_test_database(_test_database: None) -> None:
    """Bring the test database to the latest schema once per session."""
    os.environ["DB_URL"] = TEST_DB_URL
    get_api_settings.cache_clear()
    try:
        command.upgrade(alembic_config(), "head")
    finally:
        del os.environ["DB_URL"]
        get_api_settings.cache_clear()
