import json

import httpx
import pytest
from webkiln_agents.search import (
    CachedSearch,
    SearchResult,
    TavilySearch,
    cache_key,
    parse_tavily,
)


def result(n: int = 1) -> SearchResult:
    return SearchResult(title=f"Site {n}", url=f"https://site{n}.example/", content="Text " * 5)


class MemoryCache:
    def __init__(self) -> None:
        self.rows: dict[str, list[SearchResult]] = {}

    async def get(self, key: str) -> list[SearchResult] | None:
        return self.rows.get(key)

    async def put(self, key: str, query: str, results: list[SearchResult]) -> None:
        self.rows[key] = results


class CountingProvider:
    def __init__(self, answer: list[SearchResult] | Exception) -> None:
        self.answer = answer
        self.queries: list[str] = []

    async def search(self, query: str) -> list[SearchResult]:
        self.queries.append(query)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_the_key_ignores_case_and_spacing() -> None:
    assert cache_key("Boutique  Hotels") == cache_key(" boutique hotels ")
    assert cache_key("boutique hotels") != cache_key("boutique motels")


def test_parsing_keeps_only_results_with_a_web_url_and_text() -> None:
    body = {
        "results": [
            {"title": " A ", "url": "https://a.example", "content": "x  y\n z"},
            {"title": "no url", "content": "text"},
            {"title": "bad scheme", "url": "javascript:alert(1)", "content": "text"},
            {"title": "no text", "url": "https://b.example", "content": "  "},
            "not a dict",
        ]
    }
    assert parse_tavily(body) == [SearchResult(title="A", url="https://a.example", content="x y z")]
    assert parse_tavily({"unexpected": 1}) == []
    assert parse_tavily(None) == []


def test_parsing_cuts_long_text_and_long_lists() -> None:
    body = {"results": [{"url": f"https://s{i}.example", "content": "w" * 5000} for i in range(9)]}
    parsed = parse_tavily(body)
    assert len(parsed) == 5
    assert all(len(r.content) == 600 for r in parsed)


async def test_tavily_sends_a_bearer_key_and_a_basic_search() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": [{"url": "https://a.example", "content": "t"}]})

    search = TavilySearch("tvly-secret", transport=httpx.MockTransport(handler))
    results = await search.search("boutique hotels")
    assert [r.url for r in results] == ["https://a.example"]
    request = seen[0]
    assert request.headers["authorization"] == "Bearer tvly-secret"
    body = json.loads(request.content)
    assert body["query"] == "boutique hotels"
    assert body["search_depth"] == "basic"
    assert body["include_answer"] is False


async def test_tavily_raises_on_an_error_status() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"detail": "bad key"}))
    with pytest.raises(httpx.HTTPStatusError):
        await TavilySearch("wrong", transport=transport).search("anything")


async def test_a_repeated_query_is_answered_from_the_cache() -> None:
    provider, cache = CountingProvider([result()]), MemoryCache()
    search = CachedSearch(provider, cache)
    first = await search.search("Boutique hotels")
    again = await search.search("boutique   hotels")
    assert first == again == [result()]
    assert provider.queries == ["Boutique hotels"]


async def test_an_empty_answer_is_not_cached() -> None:
    provider, cache = CountingProvider([]), MemoryCache()
    search = CachedSearch(provider, cache)
    await search.search("nothing here")
    await search.search("nothing here")
    assert len(provider.queries) == 2
    assert not cache.rows


async def test_a_broken_cache_does_not_stop_the_search() -> None:
    class BrokenCache:
        async def get(self, key: str) -> list[SearchResult] | None:
            msg = "db down"
            raise ConnectionError(msg)

        async def put(self, key: str, query: str, results: list[SearchResult]) -> None:
            msg = "db down"
            raise ConnectionError(msg)

    provider = CountingProvider([result()])
    assert await CachedSearch(provider, BrokenCache()).search("q") == [result()]


async def test_a_provider_error_reaches_the_caller() -> None:
    provider = CountingProvider(httpx.ConnectError("offline"))
    with pytest.raises(httpx.ConnectError):
        await CachedSearch(provider, MemoryCache()).search("q")
