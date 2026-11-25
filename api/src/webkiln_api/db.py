from sqlalchemy import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from webkiln_api.config import get_api_settings


def sqlalchemy_url() -> URL:
    """The configured database URL, set to use the psycopg 3 driver."""
    return make_url(get_api_settings().DB_URL).set(drivername="postgresql+psycopg")


def create_engine() -> AsyncEngine:
    return create_async_engine(sqlalchemy_url())


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)
