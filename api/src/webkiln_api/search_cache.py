"""The Postgres cache behind web search."""

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from webkiln_agents.search import SearchResult

from webkiln_api.store import Sessions
from webkiln_api.tables import SearchCacheRow

# Competitor pages change slowly. A week keeps a demo free of repeat credits and stays fresh.
CACHE_TTL = timedelta(days=7)


class PostgresSearchCache:
    def __init__(self, sessions: Sessions, *, ttl: timedelta = CACHE_TTL) -> None:
        self._sessions = sessions
        self._ttl = ttl

    async def get(self, key: str) -> list[SearchResult] | None:
        async with self._sessions() as session:
            rows = await session.scalar(
                select(SearchCacheRow.results).where(
                    SearchCacheRow.key == key,
                    SearchCacheRow.created_at > func.now() - self._ttl,
                )
            )
        return None if rows is None else [SearchResult.model_validate(row) for row in rows]

    async def put(self, key: str, query: str, results: list[SearchResult]) -> None:
        values = {"key": key, "query": query, "results": [r.model_dump() for r in results]}
        statement = insert(SearchCacheRow).values(values)
        async with self._sessions.begin() as session:
            await session.execute(
                statement.on_conflict_do_update(
                    index_elements=["key"],
                    set_={"results": statement.excluded.results, "created_at": func.now()},
                )
            )
