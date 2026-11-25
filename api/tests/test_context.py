import pytest
from fastapi.testclient import TestClient
from webkiln_agents.core.config import Settings
from webkiln_agents.manager.context import Context
from webkiln_agents.models import ChainLLM, ConfigError
from webkiln_agents.search import CachedSearch
from webkiln_agents.testing import SAMPLE_BRIEF
from webkiln_agents.workers.base import WorkerInput
from webkiln_api import context as api_context
from webkiln_api.app import create_app
from webkiln_api.context import build_context
from webkiln_api.db import create_engine, create_session_factory

KEY_SETTINGS = ("GOOGLE_API_KEYS", "GROQ_API_KEYS", "OPEN_ROUTER_API_KEYS", "TAVILY_API_KEY")


def use_settings(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    """Settings that read no `.env` file and hold only `values`."""
    for name in KEY_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(api_context, "get_settings", lambda: Settings(_env_file=None, **values))  # pyrefly: ignore[bad-argument-type]


@pytest.fixture
def sessions():
    return create_session_factory(create_engine())


def test_a_real_run_gets_the_model_provider(
    monkeypatch: pytest.MonkeyPatch, sessions: object
) -> None:
    use_settings(monkeypatch)
    context = build_context(sessions)  # pyrefly: ignore[bad-argument-type]
    assert isinstance(context.llm, ChainLLM)
    assert {"niche", "audience", "seo"} <= set(context.workers)


def test_search_is_on_only_with_a_tavily_key(
    monkeypatch: pytest.MonkeyPatch, sessions: object
) -> None:
    use_settings(monkeypatch)
    assert build_context(sessions).search is None  # pyrefly: ignore[bad-argument-type]
    use_settings(monkeypatch, TAVILY_API_KEY="  ")  # a blank value in .env is no key
    assert build_context(sessions).search is None  # pyrefly: ignore[bad-argument-type]
    use_settings(monkeypatch, TAVILY_API_KEY="tvly-test")
    assert isinstance(build_context(sessions).search, CachedSearch)  # pyrefly: ignore[bad-argument-type]


async def test_a_real_worker_names_the_missing_key_instead_of_failing_obscurely(
    monkeypatch: pytest.MonkeyPatch, sessions: object
) -> None:
    """The wiring check: the worker the API registers reaches the model chain."""
    use_settings(monkeypatch)
    context: Context = build_context(sessions)  # pyrefly: ignore[bad-argument-type]
    task = {"id": "niche", "agent": "niche", "instruction": "", "depends_on": []}
    packet: WorkerInput = {
        "task": task,
        "brief": SAMPLE_BRIEF,
        "analysis": None,
        "context": {},
        "previous": None,
    }
    with pytest.raises(ConfigError, match="none of those providers has an API key"):
        await context.workers["niche"](packet, context)


def test_the_app_builds_the_real_context_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    use_settings(monkeypatch)
    app = create_app()
    with TestClient(app):
        assert isinstance(app.state.context.llm, ChainLLM)
        assert app.state.supervisor.context is app.state.context
