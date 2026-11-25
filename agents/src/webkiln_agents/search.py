"""Web search for the research workers: a provider, a cache in front of it, and the result type.

Search results are untrusted text. The workers put them in a tagged block of the prompt and tell
the model never to follow instructions found there.
"""

import hashlib
import logging
from typing import Protocol

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
RESULTS_PER_QUERY = 5
SNIPPET_LENGTH = 600  # characters of page text kept per result
SEARCH_TIMEOUT = 20.0  # seconds


class SearchResult(BaseModel):
    title: str
    url: str
    content: str


class SearchProvider(Protocol):
    async def search(self, query: str) -> list[SearchResult]:
        """The results for one query. Raises when the search service cannot answer."""
        ...


class SearchCache(Protocol):
    """Stores results by a hash of the query, so a re-run spends no search credits."""

    async def get(self, key: str) -> list[SearchResult] | None: ...

    async def put(self, key: str, query: str, results: list[SearchResult]) -> None: ...


def cache_key(query: str) -> str:
    """The same key for queries that differ only in case or spacing."""
    normal = " ".join(query.lower().split())
    return hashlib.sha256(f"tavily:basic:{RESULTS_PER_QUERY}:{normal}".encode()).hexdigest()


def parse_tavily(body: object) -> list[SearchResult]:
    """The usable results of a Tavily answer. An entry without a web URL or text is dropped."""
    items = body.get("results") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return []
    results: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url, content = item.get("url"), item.get("content")
        if not (isinstance(url, str) and url.startswith(("http://", "https://"))):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        title = item.get("title")
        results.append(
            SearchResult(
                title=title.strip() if isinstance(title, str) else "",
                url=url,
                content=" ".join(content.split())[:SNIPPET_LENGTH],
            )
        )
    return results[:RESULTS_PER_QUERY]


class TavilySearch:
    """Tavily's search API. One basic search costs one credit."""

    def __init__(self, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._api_key = api_key
        self._transport = transport

    async def search(self, query: str) -> list[SearchResult]:
        payload = {
            "query": query[:400],
            "search_depth": "basic",
            "max_results": RESULTS_PER_QUERY,
            "include_answer": False,
        }
        async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT, transport=self._transport) as client:
            response = await client.post(
                TAVILY_URL, json=payload, headers={"Authorization": f"Bearer {self._api_key}"}
            )
            response.raise_for_status()
            return parse_tavily(response.json())


class CachedSearch:
    """Answers from the cache when it can and asks the provider otherwise.

    A broken cache never stops a search: its errors are logged by type and the provider answers.
    An empty answer is not stored, so a query that found nothing is tried again next time.
    """

    def __init__(self, provider: SearchProvider, cache: SearchCache) -> None:
        self._provider = provider
        self._cache = cache

    async def search(self, query: str) -> list[SearchResult]:
        key = cache_key(query)
        try:
            cached = await self._cache.get(key)
        except Exception as error:
            logger.warning("search cache read failed: %s", type(error).__name__)
            cached = None
        if cached is not None:
            return cached
        results = await self._provider.search(query)
        if results:
            try:
                await self._cache.put(key, query, results)
            except Exception as error:
                logger.warning("search cache write failed: %s", type(error).__name__)
        return results
