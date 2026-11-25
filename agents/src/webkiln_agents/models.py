"""The model chains. One chain per role, tried in order until a model answers.

Only real providers live here. Agents get models through `LLMProvider` (see llm.py), so tests
use a `FakeLLM` and never reach this module.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, SecretStr

from webkiln_agents.core.config import Settings
from webkiln_agents.llm import StructuredCall

logger = logging.getLogger(__name__)

type Provider = Literal["google", "groq", "openrouter"]
PROVIDERS: tuple[Provider, ...] = ("google", "groq", "openrouter")

# How many calls a role may spend in one full build is written next to its chain. A full build
# is 8 to 13 calls (plan 06, section 6.3). The free tiers are small, so these are budgets, not
# limits: the Manager runs only when something needs a decision.
#
# Gemini carries the workers, because its free tokens-per-minute are the largest. Groq's
# `gpt-oss-120b` allows about 8,000 tokens a minute, so it is only ever a last resort for a
# worker. Free-tier model IDs change often: set `MODEL_<ROLE>` in `.env` when one stops working.
GEMINI_CHAIN = "google:gemini-2.5-flash,google:gemini-2.5-flash-lite,groq:openai/gpt-oss-120b"
DEFAULT_CHAINS: dict[str, str] = {
    # 0 to 2 calls: the brief analysis, and one decision per wave that had a failure.
    "manager": "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free,groq:openai/gpt-oss-120b",
    "niche": GEMINI_CHAIN,  # 1 call
    "audience": GEMINI_CHAIN,  # 1 call
    "competitor": GEMINI_CHAIN,  # 1 call, plus up to 3 searches (not model calls)
    "seo": GEMINI_CHAIN,  # 1 call
    "brand": GEMINI_CHAIN,  # 1 call, plus 1 for every round of feedback that reworks it
}

# Requests a minute each provider's free tier allows, shared by every model of that provider.
REQUESTS_PER_MINUTE: dict[Provider, int] = {"google": 10, "groq": 30, "openrouter": 20}

# `ChatOpenRouter` has no JSON mode, and the Nemotron page says `response_format` is not
# supported there, so OpenRouter uses tool calling. `just check-structured-output` tests these.
METHODS: dict[Provider, Literal["function_calling", "json_schema"]] = {
    "google": "json_schema",
    "groq": "json_schema",
    "openrouter": "function_calling",
}

REQUEST_TIMEOUT = 120.0  # seconds; the graph's own timeout also covers rate-limit waits


class ConfigError(Exception):
    """The settings cannot run this role. It is not retried, and its text reaches the user."""


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def parse_chain(text: str) -> list[ModelSpec]:
    """`provider:model,provider:model` to specs. A model ID may hold colons (`...:free`)."""
    specs: list[ModelSpec] = []
    for part in filter(None, (piece.strip() for piece in text.split(","))):
        provider, _, model = part.partition(":")
        if provider not in PROVIDERS or not model:
            msg = f"Bad model {part!r}: write it as provider:model with a provider of {PROVIDERS}"
            raise ConfigError(msg)
        specs.append(ModelSpec(provider, model))
    return specs


def chain_for(role: str, settings: Settings) -> list[ModelSpec]:
    """The role's models that have a key, in order."""
    override: str | None = getattr(settings, f"MODEL_{role.upper()}", None)
    text = override or DEFAULT_CHAINS.get(role)
    if text is None:
        msg = f"No models are set for the role {role!r}"
        raise ConfigError(msg)
    specs = [spec for spec in parse_chain(text) if _api_key(spec.provider, settings) is not None]
    if not specs:
        msg = (
            f"The role {role!r} uses {text!r}, but none of those providers has an API key. "
            "Add one to .env or set MODEL_" + role.upper()
        )
        raise ConfigError(msg)
    return specs


def _api_key(provider: Provider, settings: Settings) -> SecretStr | None:
    keys = {
        "google": settings.GOOGLE_API_KEYS,
        "groq": settings.GROQ_API_KEYS,
        "openrouter": settings.OPEN_ROUTER_API_KEYS,
    }[provider]
    # Extra keys add no quota: the providers limit per account, so only the first is used.
    return keys[0] if keys else None


_limiters: dict[Provider, InMemoryRateLimiter] = {}


def _limiter(provider: Provider) -> InMemoryRateLimiter:
    """One limiter per provider in this process, so parallel workers share the request budget."""
    if provider not in _limiters:
        _limiters[provider] = InMemoryRateLimiter(
            requests_per_second=REQUESTS_PER_MINUTE[provider] / 60,
            max_bucket_size=4,  # the four research workers may start together
        )
    return _limiters[provider]


def build_model(spec: ModelSpec, settings: Settings) -> BaseChatModel:
    """One chat model. SDK retries are off: the chain and the graph's retry policy own them."""
    key = _api_key(spec.provider, settings)
    common = {"api_key": key, "max_retries": 0, "rate_limiter": _limiter(spec.provider)}
    if spec.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=spec.model, timeout=REQUEST_TIMEOUT, **common)  # pyrefly: ignore[bad-argument-type]
    if spec.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=spec.model, timeout=REQUEST_TIMEOUT, **common)  # pyrefly: ignore[bad-argument-type]
    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(model=spec.model, timeout=int(REQUEST_TIMEOUT * 1000), **common)  # pyrefly: ignore[bad-argument-type]


type ModelFactory = Callable[[ModelSpec, Settings], BaseChatModel]


class _Chain[T: BaseModel]:
    """Calls each model of a role in turn and returns the first valid answer.

    A model that raises (rate limit, outage, output that does not parse) or does not comply
    (`None`) passes the same prompt to the next one. When all fail, the last error is raised so
    the graph can tell a transient one from a permanent one, and if no model raised the answer is
    `None`, which the worker treats as a failed attempt.
    """

    def __init__(self, calls: list[tuple[ModelSpec, StructuredCall[T]]]) -> None:
        self._calls = calls

    async def ainvoke(self, input: list[BaseMessage]) -> T | None:
        last_error: Exception | None = None
        for spec, call in self._calls:
            try:
                result = await call.ainvoke(input)
            except Exception as error:
                # Only the type is logged: a message could hold parts of the prompt.
                logger.warning("model %s failed: %s", spec, type(error).__name__)
                last_error = error
                continue
            if result is not None:
                return result
            logger.warning("model %s gave no valid answer", spec)
        if last_error is not None:
            raise last_error
        return None


class ChainLLM:
    """The real `LLMProvider`: the chains of the settings, with structured output on each model."""

    def __init__(self, settings: Settings, *, factory: ModelFactory = build_model) -> None:
        self._settings = settings
        self._factory = factory
        self._models: dict[ModelSpec, BaseChatModel] = {}

    def _model(self, spec: ModelSpec) -> BaseChatModel:
        if spec not in self._models:
            self._models[spec] = self._factory(spec, self._settings)
        return self._models[spec]

    def structured[T: BaseModel](self, role: str, schema: type[T]) -> StructuredCall[T]:
        """The chain for `role`. A bad setting raises `ConfigError` here, before any call."""
        calls: list[tuple[ModelSpec, StructuredCall[T]]] = [
            (
                spec,
                cast(
                    StructuredCall[T],
                    self._model(spec).with_structured_output(schema, method=METHODS[spec.provider]),
                ),
            )
            for spec in chain_for(role, self._settings)
        ]
        return _Chain(calls)
