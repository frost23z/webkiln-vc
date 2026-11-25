from datetime import timedelta

from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy import text
from webkiln_agents.search import SearchResult
from webkiln_api.db import create_engine, create_session_factory, sqlalchemy_url
from webkiln_api.search_cache import PostgresSearchCache

RESULTS = [SearchResult(title="A", url="https://a.example", content="text")]


def forget(key: str) -> None:
    engine = create_sync_engine(sqlalchemy_url())
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM search_cache WHERE key = :key"), {"key": key})
    engine.dispose()


def age(key: str, days: int) -> None:
    engine = create_sync_engine(sqlalchemy_url())
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE search_cache SET created_at = now() - make_interval(days => :d)"
                " WHERE key = :key"
            ),
            {"d": days, "key": key},
        )
    engine.dispose()


async def test_results_are_stored_and_read_back() -> None:
    key = "test-store"
    forget(key)
    engine = create_engine()
    try:
        cache = PostgresSearchCache(create_session_factory(engine))
        assert await cache.get(key) is None
        await cache.put(key, "boutique hotels", RESULTS)
        assert await cache.get(key) == RESULTS
    finally:
        await engine.dispose()
        forget(key)


async def test_a_second_write_replaces_the_first() -> None:
    key = "test-replace"
    forget(key)
    engine = create_engine()
    try:
        cache = PostgresSearchCache(create_session_factory(engine))
        await cache.put(key, "q", RESULTS)
        newer = [SearchResult(title="B", url="https://b.example", content="newer")]
        await cache.put(key, "q", newer)
        assert await cache.get(key) == newer
    finally:
        await engine.dispose()
        forget(key)


async def test_old_results_are_not_served() -> None:
    key = "test-ttl"
    forget(key)
    engine = create_engine()
    try:
        cache = PostgresSearchCache(create_session_factory(engine), ttl=timedelta(days=7))
        await cache.put(key, "q", RESULTS)
        age(key, 8)
        assert await cache.get(key) is None
        age(key, 6)
        assert await cache.get(key) == RESULTS
    finally:
        await engine.dispose()
        forget(key)
