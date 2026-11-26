"""The embedding model. One model for everything: changing it means embedding everything again."""

import math
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMS = 768  # `output_dimensionality`; the pgvector HNSW index allows up to 2,000


def unit(vector: Sequence[float]) -> list[float]:
    """The vector scaled to length 1. Cut-down embeddings are not normalised by the model."""
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector] if length else list(vector)


class NormalisedEmbeddings(Embeddings):
    """Wraps any embeddings class and returns unit vectors."""

    def __init__(self, inner: Embeddings) -> None:
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [unit(v) for v in self._inner.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return unit(self._inner.embed_query(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [unit(v) for v in await self._inner.aembed_documents(texts)]

    async def aembed_query(self, text: str) -> list[float]:
        return unit(await self._inner.aembed_query(text))


def google_embeddings(api_key: str) -> Embeddings:
    inner = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=api_key, output_dimensionality=EMBEDDING_DIMS
    )
    return NormalisedEmbeddings(inner)
