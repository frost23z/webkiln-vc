from webkiln_agents.core.config import get_settings
from webkiln_agents.manager.context import Context
from webkiln_agents.models import ChainLLM
from webkiln_agents.search import CachedSearch, SearchProvider, TavilySearch
from webkiln_agents.workers.registry import WORKERS

from webkiln_api.search_cache import PostgresSearchCache
from webkiln_api.store import Sessions


def build_context(sessions: Sessions) -> Context:
    """The context of a real run: the registered workers, the model chains and web search.

    Nothing is called here. A missing model key shows up when a worker asks for its model, and
    then the task fails with a message that names the key. Without a Tavily key there is no
    search, and the competitor report says so. Tests pass their own context to `create_app`.
    """
    settings = get_settings()
    search: SearchProvider | None = None
    tavily_key = (
        settings.TAVILY_API_KEY.get_secret_value().strip() if settings.TAVILY_API_KEY else ""
    )
    if tavily_key:
        search = CachedSearch(TavilySearch(tavily_key), PostgresSearchCache(sessions))
    return Context(workers=dict(WORKERS), llm=ChainLLM(settings), search=search)
