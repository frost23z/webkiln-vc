import json
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from support import BRIEF, api, approve, run_to_review, saved_run, with_db
from webkiln_agents.manager.context import Context
from webkiln_agents.site.checks import check_site
from webkiln_agents.testing import (
    SAMPLE_BRAND,
    FakeSiteLLM,
    fake_context,
    fake_workers,
    sample_wireframe,
)
from webkiln_agents.workers.base import WorkerResult
from webkiln_agents.workers.generate import generate_worker
from webkiln_api import store
from webkiln_api.app import create_app
from webkiln_api.checkpoint import serde
from webkiln_api.site import is_site_output, strip_update, without_files
from webkiln_api.tables import Run, SiteFile, SiteVersion

WIREFRAME = sample_wireframe()


async def wireframe_worker(*_: Any) -> WorkerResult:
    return WorkerResult(summary="wireframe", data=WIREFRAME.model_dump(mode="json"))


async def brand_worker(*_: Any) -> WorkerResult:
    return WorkerResult(summary="brand", data=SAMPLE_BRAND)


def site_context() -> Context:
    """Fake research, brand and wireframe, and the real generate subgraph on a fake model."""
    workers = {
        **fake_workers(),
        "brand": brand_worker,
        "wireframe": wireframe_worker,
        "generate": generate_worker,
    }
    return fake_context(workers=workers, llm=FakeSiteLLM())


def saved_site(thread_id: str) -> tuple[list[SiteVersion], dict[str, str]]:
    def read(session: Session) -> tuple[list[SiteVersion], dict[str, str]]:
        run = session.scalars(select(Run).where(Run.thread_id == thread_id)).one()
        versions = list(session.scalars(select(SiteVersion).where(SiteVersion.run_id == run.id)))
        files = {
            f.path: f.content
            for f in session.scalars(select(SiteFile).where(SiteFile.run_id == run.id))
        }
        return versions, files

    return with_db(read)


def build_site_run(client) -> str:
    thread_id = run_to_review(client, BRIEF)
    events = approve(client, thread_id)
    assert events[-1] == ("done", {"status": "completed"})
    return thread_id


def test_the_finished_run_saves_site_version_1_with_every_file() -> None:
    with api(site_context()) as client:
        thread_id = build_site_run(client)

    versions, files = saved_site(thread_id)
    assert [(v.version, v.parent_version, v.cause) for v in versions] == [(1, None, "generate")]
    assert versions[0].wireframe["navigation"] == WIREFRAME.navigation
    assert set(versions[0].copies) == {"index", "rooms", "offers", "contact"}
    assert {"index.html", "rooms.html", "assets/tokens.css", "assets/base.css"} <= set(files)
    assert check_site(files, WIREFRAME) == []


def test_the_stream_and_the_run_detail_carry_paths_not_files() -> None:
    with api(site_context()) as client:
        thread_id = run_to_review(client, BRIEF)
        events = approve(client, thread_id)
        report = client.get(f"/runs/{thread_id}/tasks/generate").json()
        detail = client.get(f"/runs/{thread_id}").text

    updates = [data for name, data in events if name == "updates"]
    sent = next(d["run_generate"]["outputs"]["generate"] for d in updates if "run_generate" in d)
    assert "files" not in sent
    assert "index.html" in sent["file_paths"]
    assert report["output"]["file_paths"] == sent["file_paths"]
    assert "files" not in report["output"]
    assert "<!doctype html>" not in detail.lower()
    assert "<!doctype html>" not in json.dumps(events).lower()


def test_the_subgraph_progress_reaches_the_run_stream() -> None:
    with api(site_context()) as client:
        thread_id = run_to_review(client, BRIEF)
        events = approve(client, thread_id)
    pages = {
        d["page"] for name, d in events if name == "custom" and d.get("event") == "page_written"
    }
    assert pages == {"index", "rooms", "offers", "contact"}


def test_saving_the_same_site_again_changes_nothing() -> None:
    app = create_app(site_context())
    with TestClient(app) as client:
        portal = client.portal
        assert portal is not None
        thread_id = build_site_run(client)
        before = saved_site(thread_id)
        portal.call(app.state.supervisor.sync, thread_id, "completed")
        after = saved_site(thread_id)
        read_back = portal.call(store.site_files, app.state.sessions, thread_id)
    assert [v.version for v in after[0]] == [1]
    assert after[1] == before[1]
    assert read_back == before[1]  # what the preview will serve is what was built


def test_the_run_without_a_site_saves_no_version() -> None:
    with api() as client:  # fake workers: `generate` returns no files
        thread_id = build_site_run(client)
    versions, files = saved_site(thread_id)
    assert versions == []
    assert files == {}
    _, _, outputs = saved_run(thread_id)
    assert {o.task_id for o in outputs} >= {"generate"}


def test_without_files_replaces_the_files_with_their_paths() -> None:
    build = {"files": {"b.html": "<p>b</p>", "a.html": "<p>a</p>"}, "pages": ["a"], "copies": {}}
    assert is_site_output(build)
    assert without_files(build) == {
        "pages": ["a"],
        "copies": {},
        "file_paths": ["a.html", "b.html"],
    }
    other = {"agent": "niche"}
    assert without_files(other) == other
    assert strip_update({"run_worker": None}) == {"run_worker": None}
    assert strip_update({"x": {"outputs": {"niche": other}}}) == {
        "x": {"outputs": {"niche": other}}
    }


def test_the_checkpoint_can_read_the_subgraph_state_back() -> None:
    from webkiln_agents.workers.generate import GenState, PageJob

    state = GenState(
        task_id="generate",
        instruction="",
        brief={"brand_name": "A"},
        analysis=None,
        brand={},
        wireframe={},
    )
    job = PageJob(**state.model_dump(), slug="index")
    for value in (state, job):
        restored = serde.loads_typed(serde.dumps_typed(value))
        assert type(restored) is type(value)
        assert restored == value
