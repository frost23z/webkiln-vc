from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, SecretStr
from webkiln_agents.core.config import Settings
from webkiln_agents.models import (
    DEFAULT_CHAINS,
    ChainLLM,
    ConfigError,
    ModelSpec,
    build_model,
    chain_for,
    parse_chain,
)


class Answer(BaseModel):
    text: str


def settings(**keys: Any) -> Settings:
    names = {
        "google": "GOOGLE_API_KEYS",
        "groq": "GROQ_API_KEYS",
        "openrouter": "OPEN_ROUTER_API_KEYS",
    }
    values = {names[provider]: [SecretStr("key")] for provider in keys}
    return Settings(_env_file=None, **values)  # pyrefly: ignore[bad-argument-type]


def test_parse_keeps_colons_inside_the_model_id() -> None:
    assert parse_chain("openrouter:nvidia/x:free, groq:openai/y") == [
        ModelSpec("openrouter", "nvidia/x:free"),
        ModelSpec("groq", "openai/y"),
    ]


@pytest.mark.parametrize("text", ["gemini", "azure:gpt", "google:"])
def test_parse_refuses_a_bad_model(text: str) -> None:
    with pytest.raises(ConfigError):
        parse_chain(text)


def test_every_default_chain_parses() -> None:
    for text in DEFAULT_CHAINS.values():
        assert parse_chain(text)


def test_chain_skips_providers_without_a_key() -> None:
    specs = chain_for("niche", settings(groq=1))
    assert [spec.provider for spec in specs] == ["groq"]


def test_chain_without_any_key_says_what_to_set() -> None:
    with pytest.raises(ConfigError, match="MODEL_NICHE"):
        chain_for("niche", settings())


def test_env_setting_overrides_the_default() -> None:
    config = Settings(_env_file=None, MODEL_SEO="groq:a,groq:b", GROQ_API_KEYS=[SecretStr("k")])  # pyrefly: ignore[bad-argument-type]
    assert [spec.model for spec in chain_for("seo", config)] == ["a", "b"]


def test_unknown_role_is_refused() -> None:
    with pytest.raises(ConfigError):
        chain_for("nobody", settings(groq=1))


def test_real_clients_build_without_network() -> None:
    config = settings(google=1, groq=1, openrouter=1)
    for text in ("google:gemini-3.5-flash-lite", "groq:openai/gpt-oss-120b", "openrouter:a/b:free"):
        (spec,) = parse_chain(text)
        assert build_model(spec, config) is not None


class FakeRunnable:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls = 0

    async def ainvoke(self, _: list) -> Any:
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeChat:
    def __init__(self, outcome: Any) -> None:
        self.runnable = FakeRunnable(outcome)

    def with_structured_output(self, schema: type, method: str) -> FakeRunnable:
        assert method in {"json_schema", "function_calling"}
        return self.runnable


def chain(*outcomes: Any) -> tuple[Any, list[FakeChat]]:
    chats = [FakeChat(outcome) for outcome in outcomes]
    by_model = {f"m{i}": chat for i, chat in enumerate(chats)}
    config = Settings(
        _env_file=None,  # pyrefly: ignore[bad-argument-type]
        GROQ_API_KEYS=[SecretStr("k")],
        MODEL_NICHE=",".join(f"groq:{name}" for name in by_model),
    )
    llm = ChainLLM(config, factory=lambda spec, _: by_model[spec.model])  # pyrefly: ignore[bad-argument-type]
    return llm.structured("niche", Answer), chats


MESSAGES = [HumanMessage("hi")]


async def test_first_good_answer_stops_the_chain() -> None:
    call, chats = chain(Answer(text="a"), Answer(text="b"))
    assert await call.ainvoke(MESSAGES) == Answer(text="a")
    assert chats[1].runnable.calls == 0


async def test_an_error_or_no_answer_moves_to_the_next_model() -> None:
    call, chats = chain(RuntimeError("429"), None, Answer(text="c"))
    assert await call.ainvoke(MESSAGES) == Answer(text="c")
    assert [chat.runnable.calls for chat in chats] == [1, 1, 1]


async def test_when_every_model_fails_the_last_error_is_raised() -> None:
    call, _ = chain(ValueError("first"), None, TimeoutError("last"))
    with pytest.raises(TimeoutError):
        await call.ainvoke(MESSAGES)


async def test_when_no_model_complies_the_answer_is_none() -> None:
    call, _ = chain(None, None)
    assert await call.ainvoke(MESSAGES) is None
