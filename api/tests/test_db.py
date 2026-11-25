import asyncio

from sqlalchemy import text
from webkiln_api.db import create_engine, sqlalchemy_url


def test_url_uses_the_psycopg_driver_and_keeps_the_rest() -> None:
    url = sqlalchemy_url()
    assert url.drivername == "postgresql+psycopg"
    assert url.database == "webkiln_test"  # the test fixture points DB_URL at the test database


def test_engine_can_run_a_query() -> None:
    async def query() -> int:
        engine = create_engine()
        try:
            async with engine.connect() as conn:
                return (await conn.execute(text("SELECT 1"))).scalar_one()
        finally:
            await engine.dispose()

    assert asyncio.run(query()) == 1
