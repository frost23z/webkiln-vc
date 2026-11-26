"""Long-term memory in the API: the embeddings, the Postgres store and the per-run handle."""

from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from langgraph.store.base import BaseStore
from langgraph.store.postgres.base import PostgresIndexConfig
from webkiln_agents.core.config import get_settings
from webkiln_agents.memory.embeddings import EMBEDDING_DIMS, google_embeddings
from webkiln_agents.memory.service import Memory

# Makes the `Memory` of one user in one project. The user and the project are in every namespace.
type MemoryFactory = Callable[[str, str], Memory]


def embeddings_from_settings() -> Embeddings | None:
    """Google embeddings with the first usable key, or `None` when there is no key. Without
    embeddings there is no memory store, and runs work without memory."""
    for key in get_settings().GOOGLE_API_KEYS:
        value = key.get_secret_value().strip()
        if value:
            return google_embeddings(value)
    return None


def index_config(embeddings: Embeddings) -> PostgresIndexConfig:
    """Vector index of the store: cosine distance over each record's `text`."""
    return {
        "dims": EMBEDDING_DIMS,
        "embed": embeddings,
        "fields": ["text"],
        "distance_type": "cosine",
    }


def memory_factory(store: BaseStore) -> MemoryFactory:
    return lambda user_id, project_id: Memory(store, user_id, project_id)
