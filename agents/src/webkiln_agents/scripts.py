"""Command line checks. They call real models, so each one spends a little quota."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from webkiln_agents.core.config import get_settings
from webkiln_agents.memory.embeddings import EMBEDDING_DIMS, EMBEDDING_MODEL, google_embeddings
from webkiln_agents.memory.service import SUPERSEDE_ABOVE
from webkiln_agents.models import (
    DEFAULT_CHAINS,
    METHODS,
    PROVIDERS,
    ConfigError,
    ModelSpec,
    _api_key,
    build_model,
    parse_chain,
)


class Ping(BaseModel):
    word: str
    number: int


async def _try(spec: ModelSpec, method: str) -> str:
    settings = get_settings()
    model = build_model(spec, settings).with_structured_output(Ping, method=method)  # pyrefly: ignore[no-matching-overload]
    try:
        result = await model.ainvoke([HumanMessage("Answer with the word 'ok' and the number 7.")])
    except Exception as error:
        return f"FAILED  {type(error).__name__}: {str(error)[:300]}"
    return "works" if isinstance(result, Ping) else f"FAILED  answered {result!r}"


async def _check() -> None:
    settings = get_settings()
    chosen: dict[str, ModelSpec] = {}
    for text in DEFAULT_CHAINS.values():  # the first model of each provider, 6 calls in all
        for spec in parse_chain(text):
            if _api_key(spec.provider, settings) is not None:
                chosen.setdefault(spec.provider, spec)
    for provider in PROVIDERS:
        if provider not in chosen:
            print(f"{provider}: no API key in .env, skipped")
    for spec in chosen.values():
        for method in ("function_calling", "json_schema"):
            used = "  <- used by the app" if METHODS[spec.provider] == method else ""
            print(f"{spec} [{method}]: {await _try(spec, method)}{used}")


def check_structured_output() -> None:
    """Try each provider's first model with both structured output methods and print the result.

    The method the app uses is `METHODS` in models.py. If it fails here and the other works,
    change it there.
    """
    try:
        asyncio.run(_check())
    except ConfigError as error:
        raise SystemExit(str(error)) from error


@dataclass(frozen=True)
class Pair:
    """Two sentences, and whether a user would call the second an updated version of the first."""

    first: str
    second: str
    same: bool


# What the supersede threshold has to tell apart. A preference reworded or extended should replace
# the old one. A different preference, even on a nearby topic, must be kept beside it.
PAIRS = (
    Pair(
        "Never use stock photos, show real photos of the place and the people instead",
        "Never use stock photos, show real photos of the place and the people or illustrations",
        same=True,
    ),
    Pair(
        "Never use stock photos, show real photos of the place and the people instead",
        "Do not use stock photography on the website, only our own real photos",
        same=True,
    ),
    Pair(
        "Keep every heading under six words",
        "Headings must be shorter than six words",
        same=True,
    ),
    Pair(
        "Never use stock photos, show real photos of the place and the people instead",
        "Keep every heading under six words",
        same=False,
    ),
    Pair(
        "Use a warm and friendly tone of voice",
        "Always show the booking button in the header",
        same=False,
    ),
    Pair(
        "Brand colours are deep green and gold with serif headings",
        "Never use stock photos, show real photos of the place and the people instead",
        same=False,
    ),
)


@dataclass(frozen=True)
class PairResult:
    pair: Pair
    similarity: float

    @property
    def supersedes(self) -> bool:
        return self.similarity >= SUPERSEDE_ABOVE

    @property
    def as_expected(self) -> bool:
        return self.supersedes == self.pair.same


async def compare_pairs(
    embeddings: Embeddings, pairs: Sequence[Pair] = PAIRS
) -> tuple[list[PairResult], list[float]]:
    """Each pair's cosine similarity, and the embedding of the first text (to check its size).
    The vectors have length 1, so the cosine is their dot product."""
    texts = [text for pair in pairs for text in (pair.first, pair.second)]
    vectors = await embeddings.aembed_documents(texts)
    results = [
        PairResult(
            pair, sum(a * b for a, b in zip(vectors[2 * i], vectors[2 * i + 1], strict=True))
        )
        for i, pair in enumerate(pairs)
    ]
    return results, vectors[0]


async def _check_memory() -> None:
    keys = [key.get_secret_value().strip() for key in get_settings().GOOGLE_API_KEYS]
    keys = [key for key in keys if key]
    if not keys:
        message = "GOOGLE_API_KEYS is not set in .env: the memory needs it for embeddings."
        raise SystemExit(message)
    try:
        results, vector = await compare_pairs(google_embeddings(keys[0]))
    except Exception as error:
        message = f"Embedding failed: {type(error).__name__}: {str(error)[:200]}"
        raise SystemExit(message) from error
    size = "as planned" if len(vector) == EMBEDDING_DIMS else f"NOT the planned {EMBEDDING_DIMS}"
    print(f"{EMBEDDING_MODEL}: {len(vector)} dimensions, {size}")
    print(f"A new text replaces the old one at a cosine of {SUPERSEDE_ABOVE} or more.\n")
    for result in results:
        wanted = "replace" if result.pair.same else "keep both"
        verdict = "replaces" if result.supersedes else "keeps both"
        mark = "ok      " if result.as_expected else "MISMATCH"
        print(f"{mark} {result.similarity:.3f}  {verdict:<10} (want: {wanted})")
        print(f"           {result.pair.first}\n           {result.pair.second}")
    same = [r.similarity for r in results if r.pair.same]
    different = [r.similarity for r in results if not r.pair.same]
    print(
        f"\nSame preference: lowest {min(same):.3f}. Different ones: highest {max(different):.3f}."
    )
    if min(same) > max(different):
        middle = (min(same) + max(different)) / 2
        print(
            f"A threshold between them separates the two. About {middle:.2f} would sit in the gap."
        )
    else:
        print("No threshold separates these pairs. Keep 0.9 (a miss only keeps both records).")


def check_memory() -> None:
    """Embed some pairs of sentences with the real model and print how similar each pair is.

    Run it with your Google key to tune `SUPERSEDE_ABOVE` in memory/service.py. It makes one
    embedding call and needs no database.
    """
    asyncio.run(_check_memory())
