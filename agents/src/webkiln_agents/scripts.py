"""Command line checks. They call real models, so each one spends a little quota."""

import asyncio

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from webkiln_agents.core.config import get_settings
from webkiln_agents.models import (
    DEFAULT_CHAINS,
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
        return f"FAILED  {type(error).__name__}: {str(error)[:120]}"
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
            print(f"{spec} [{method}]: {await _try(spec, method)}")


def check_structured_output() -> None:
    """Try each provider's first model with both structured output methods and print the result.

    The method the app uses is `METHODS` in models.py. If it fails here and the other works,
    change it there.
    """
    try:
        asyncio.run(_check())
    except ConfigError as error:
        raise SystemExit(str(error)) from error
