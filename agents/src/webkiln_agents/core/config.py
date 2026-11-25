from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Keys and model choices, read from `.env` (here or one folder up) and the environment.

    Every key is optional, so the app starts with only the providers you have. A role whose
    chain has no usable key fails its task with a clear message instead of crashing the app.
    Each `MODEL_*` setting overrides one role's chain: `provider:model,provider:model,...`
    with the providers `google`, `groq` and `openrouter`. The first model that answers wins.
    """

    app_name: str = "webkiln-api"
    debug: bool = False

    GOOGLE_API_KEYS: list[SecretStr] = []
    GROQ_API_KEYS: list[SecretStr] = []
    OPEN_ROUTER_API_KEYS: list[SecretStr] = []
    TAVILY_API_KEY: SecretStr | None = None

    MODEL_MANAGER: str | None = None
    MODEL_NICHE: str | None = None
    MODEL_AUDIENCE: str | None = None
    MODEL_COMPETITOR: str | None = None
    MODEL_SEO: str | None = None

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"), env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
