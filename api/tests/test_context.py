import pytest
from webkiln_agents.core.config import Settings
from webkiln_agents.manager.context import Context
from webkiln_agents.models import ChainLLM, ConfigError
from webkiln_agents.testing import SAMPLE_BRIEF
from webkiln_agents.workers.base import WorkerInput
from webkiln_api import context as api_context
from webkiln_api.context import build_context

KEY_SETTINGS = ("GOOGLE_API_KEYS", "GROQ_API_KEYS", "OPEN_ROUTER_API_KEYS")


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings that read no `.env` file and have no API keys."""
    for name in KEY_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(api_context, "get_settings", lambda: Settings(_env_file=None))


def test_a_real_run_gets_the_model_provider(no_keys: None) -> None:
    context = build_context()
    assert isinstance(context.llm, ChainLLM)
    assert {"niche", "audience", "seo"} <= set(context.workers)


async def test_a_real_worker_names_the_missing_key_instead_of_failing_obscurely(
    no_keys: None,
) -> None:
    """The wiring check: the worker the API registers reaches the model chain."""
    context: Context = build_context()
    task = {"id": "niche", "agent": "niche", "instruction": "", "depends_on": []}
    packet: WorkerInput = {"task": task, "brief": SAMPLE_BRIEF, "analysis": None, "context": {}}
    with pytest.raises(ConfigError, match="none of those providers has an API key"):
        await context.workers["niche"](packet, context)
