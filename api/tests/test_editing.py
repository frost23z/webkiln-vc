"""Chat edits, versions, restore and the preview, through the real API and graph on fake models."""

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from support import BRIEF, api, approve, finish, run_to_review, saved_run, with_db
from webkiln_agents.manager.context import Context
from webkiln_agents.testing import (
    SAMPLE_BRAND,
    FakeEditLLM,
    FakeSiteLLM,
    fake_context,
    fake_workers,
    sample_copy,
    sample_site_copy,
    sample_wireframe,
)
from webkiln_agents.workers.base import WorkerResult
from webkiln_agents.workers.generate import generate_worker
from webkiln_api.tables import ChatMessage, Run, SiteFile

WIREFRAME = sample_wireframe()
HERO = "index-hero"
HEADLINE = "Wake up to the forest"


async def wireframe_worker(*_: Any) -> WorkerResult:
    return WorkerResult(summary="wireframe", data=WIREFRAME.model_dump(mode="json"))


async def brand_worker(*_: Any) -> WorkerResult:
    return WorkerResult(summary="brand", data=SAMPLE_BRAND)


def edit_context(script: dict[str, list[Any]], site: FakeSiteLLM | None = None) -> Context:
    workers = {
        **fake_workers(),
        "brand": brand_worker,
        "wireframe": wireframe_worker,
        "generate": generate_worker,
    }
    return fake_context(workers=workers, llm=FakeEditLLM(script, site=site))


def intent(kind: str, instruction: str = "Do what the user asked.", **fields: Any) -> dict:
    return {"intent": kind, "confidence": 0.9, "instruction": instruction, **fields}


def hero_edit(*, messages: int = 1) -> dict[str, list[Any]]:
    words = sample_copy([(HERO, "hero", HEADLINE)])["sections"][0]
    return {
        "manager:EditIntent": [intent("content_edit", "Warmer heading", section=HERO)] * messages,
        "content:SectionCopy": [words] * messages,
    }


def finished_site(client: TestClient) -> str:
    thread_id = run_to_review(client, BRIEF)
    assert approve(client, thread_id)[-1] == ("done", {"status": "completed"})
    return thread_id


def say(client: TestClient, thread_id: str, text: str, **body: Any):
    return client.post(f"/runs/{thread_id}/messages", json={"message": text, **body})


def edit(client: TestClient, thread_id: str, text: str, **body: Any) -> list[tuple[str, dict]]:
    response = say(client, thread_id, text, **body)
    assert response.status_code == 202, response.text
    return finish(client, thread_id)


def page(client: TestClient, thread_id: str, version: int, path: str = "index.html"):
    return client.get(f"/preview/{thread_id}/{version}/{path}")


def test_a_chat_message_makes_version_2_and_the_preview_shows_it() -> None:
    with api(edit_context(hero_edit())) as client:
        thread_id = finished_site(client)
        events = edit(client, thread_id, "Make the hero heading warmer", base_version=1)

        assert events[-1] == ("done", {"status": "completed"})
        updated = next(d for kind, d in events if kind == "custom" and d["event"] == "site_updated")
        assert (updated["version"], updated["paths"], updated["sections"]) == (
            2,
            ["index.html"],
            [HERO],
        )
        assert HEADLINE in page(client, thread_id, 2).text
        assert HEADLINE not in page(client, thread_id, 1).text  # the old version is kept
        v1_files = {
            p: page(client, thread_id, 1, p).text for p in ("rooms.html", "assets/base.css")
        }
        for path, text in v1_files.items():  # every other file is the same in both versions
            assert page(client, thread_id, 2, path).text == text
        versions = client.get(f"/runs/{thread_id}/versions").json()
        assert [(v["version"], v["parent_version"], v["cause"]) for v in versions] == [
            (2, 1, "content_edit"),
            (1, None, "generate"),
        ]


def test_a_version_saves_only_the_files_it_changed() -> None:
    with api(edit_context(hero_edit())) as client:
        thread_id = finished_site(client)
        edit(client, thread_id, "Make the hero heading warmer")

    def read(session: Session) -> dict[int, set[str]]:
        run = session.scalars(select(Run).where(Run.thread_id == thread_id)).one()
        rows = session.scalars(select(SiteFile).where(SiteFile.run_id == run.id))
        saved: dict[int, set[str]] = {}
        for row in rows:
            saved.setdefault(row.version, set()).add(row.path)
        return saved

    saved = with_db(read)
    assert saved[2] == {"index.html"}
    assert {"index.html", "rooms.html", "assets/tokens.css"} <= saved[1]


def test_the_chat_is_saved_in_full_and_replayed_in_order() -> None:
    with api(edit_context(hero_edit())) as client:
        thread_id = finished_site(client)
        assert client.get(f"/runs/{thread_id}/messages").json() == []
        edit(client, thread_id, "Make the hero heading warmer")
        chat = client.get(f"/runs/{thread_id}/messages").json()

    assert [m["role"] for m in chat] == ["user", "assistant"]
    assert chat[0]["content"] == "Make the hero heading warmer"
    assert "version 2" in chat[1]["content"]

    def rows(session: Session) -> int:
        run = session.scalars(select(Run).where(Run.thread_id == thread_id)).one()
        return len(session.scalars(select(ChatMessage).where(ChatMessage.run_id == run.id)).all())

    assert with_db(rows) == 2


def test_neither_the_stream_nor_the_run_detail_carries_the_files_of_an_edit() -> None:
    with api(edit_context(hero_edit())) as client:
        thread_id = finished_site(client)
        say(client, thread_id, "Make the hero heading warmer")
        stream = client.get(f"/runs/{thread_id}/stream").text
        detail = client.get(f"/runs/{thread_id}").json()

    for rendered in ("<html", "<h1", "--color-primary:"):  # a page and the tokens, as files
        assert rendered not in stream
    state = detail["state"]
    assert "files" not in state["site"]
    assert "index.html" in state["site"]["file_paths"]
    assert state["site_changes"][0]["paths"] == ["index.html"]
    assert "changed" not in state["site_changes"][0]


def test_a_second_message_on_a_busy_run_is_refused_clearly() -> None:
    with api(edit_context(hero_edit())) as client:
        thread_id = finished_site(client)
        first = say(client, thread_id, "Make the hero heading warmer")
        second = say(client, thread_id, "And make it shorter")
        assert first.status_code == 202
        assert second.status_code == 409
        assert "being changed right now" in second.json()["error"]["message"]
        assert finish(client, thread_id)[-1] == ("done", {"status": "completed"})
        assert len(client.get(f"/runs/{thread_id}/versions").json()) == 2  # one edit, not two


def test_an_edit_made_on_an_old_version_is_refused_and_changes_nothing() -> None:
    with api(edit_context(hero_edit())) as client:
        thread_id = finished_site(client)
        response = say(client, thread_id, "Change the hero", base_version=7)
        assert response.status_code == 409
        assert "you saw version 7" in response.json()["error"]["message"]
        assert client.get(f"/runs/{thread_id}/messages").json() == []
        assert client.get(f"/runs/{thread_id}").json()["status"] == "completed"
        # The user reloaded, so the same message on the newest version goes through.
        assert say(client, thread_id, "Change the hero", base_version=1).status_code == 202
        finish(client, thread_id)
        stale = say(client, thread_id, "Again", base_version=1)
        assert stale.status_code == 409
        assert "now version 2" in stale.json()["error"]["message"]


def test_a_message_is_refused_before_the_site_exists() -> None:
    with api(edit_context({})) as client:
        thread_id = run_to_review(client, BRIEF)  # paused at the brand review
        response = say(client, thread_id, "Change the hero")
        assert response.status_code == 409
        assert "Approve it first" in response.json()["error"]["message"]
        assert client.get("/runs/missing/messages").status_code == 404
        assert say(client, "missing", "Hello").status_code == 404


def test_an_empty_or_huge_message_is_a_validation_error() -> None:
    with api(edit_context({})) as client:
        thread_id = finished_site(client)
        assert say(client, thread_id, "").status_code == 422
        assert say(client, thread_id, "x" * 2001).status_code == 422
        assert say(client, thread_id, "Hi", base_version=0).status_code == 422


def test_a_question_is_answered_in_the_chat_and_makes_no_version() -> None:
    script = {"manager:EditIntent": [intent("question", "Fonts?", reply="Playfair and Inter.")]}
    with api(edit_context(script)) as client:
        thread_id = finished_site(client)
        events = edit(client, thread_id, "What fonts do we use?")
        chat = client.get(f"/runs/{thread_id}/messages").json()
        versions = client.get(f"/runs/{thread_id}/versions").json()

    assert events[-1] == ("done", {"status": "completed"})
    assert chat[-1] == {"role": "assistant", "content": "Playfair and Inter."}
    assert [v["version"] for v in versions] == [1]
    assert saved_run(thread_id)[0].status == "completed"


def test_a_later_message_can_refer_to_an_earlier_one_through_the_saved_thread() -> None:
    script = hero_edit(messages=2)
    with api(edit_context(script)) as client:
        thread_id = finished_site(client)
        edit(client, thread_id, "Make the hero heading warmer")
        edit(client, thread_id, "Do the same for the rooms page")
        prompts = client.app.state.context.llm.prompts_for("manager", None)  # type: ignore[attr-defined]
        chat = client.get(f"/runs/{thread_id}/messages").json()

    assert [m["role"] for m in chat] == ["user", "assistant", "user", "assistant"]
    last = " ".join(str(m.content) for m in prompts[-1])
    assert "Make the hero heading warmer" in last  # the model read the earlier instruction


def test_restoring_version_1_makes_version_3_with_the_files_of_version_1() -> None:
    with api(edit_context(hero_edit())) as client:
        thread_id = finished_site(client)
        edit(client, thread_id, "Make the hero heading warmer")
        response = client.post(f"/runs/{thread_id}/versions/1/restore")

        assert response.status_code == 201
        assert response.json() == {"version": 3, "restored_from": 1}
        assert HEADLINE not in page(client, thread_id, 3).text
        assert HEADLINE in page(client, thread_id, 2).text  # nothing was deleted
        causes = [
            (v["version"], v["cause"]) for v in client.get(f"/runs/{thread_id}/versions").json()
        ]
        assert causes == [(3, "restore"), (2, "content_edit"), (1, "generate")]
        assert client.get(f"/runs/{thread_id}").json()["status"] == "completed"
        chat = client.get(f"/runs/{thread_id}/messages").json()
        assert chat[-1]["content"].startswith("Restored version 1 as version 3")


def test_the_restore_of_the_current_or_a_missing_version_is_refused() -> None:
    with api(edit_context(hero_edit())) as client:
        thread_id = finished_site(client)
        assert client.post(f"/runs/{thread_id}/versions/1/restore").status_code == 409
        assert client.post(f"/runs/{thread_id}/versions/9/restore").status_code == 404
        assert client.post(f"/runs/{thread_id}/versions/0/restore").status_code == 422
        assert client.get(f"/runs/{thread_id}").json()["status"] == "completed"  # lock released


def test_an_edit_after_a_restore_goes_on_from_the_restored_site() -> None:
    with api(edit_context(hero_edit(messages=2))) as client:
        thread_id = finished_site(client)
        edit(client, thread_id, "Make the hero heading warmer")
        client.post(f"/runs/{thread_id}/versions/1/restore")
        edit(client, thread_id, "Warmer again", base_version=3)
        assert [v["version"] for v in client.get(f"/runs/{thread_id}/versions").json()] == [
            4,
            3,
            2,
            1,
        ]
        assert HEADLINE in page(client, thread_id, 4).text


def test_a_restore_drops_a_page_that_a_later_version_added() -> None:
    about = sample_copy(
        [("about-hero", "hero", "Our story"), ("about-about", "about", "How it began")],
        "about boutique hotel",
    )
    plan = {
        "page": {
            "slug": "about",
            "name": "About",
            "purpose": "Tell the story of the hotel and the people who run it.",
            "seo": {
                "target_keyword": "about boutique hotel",
                "title": "About Cedar Stay",
                "meta_description": "The story of Cedar Stay, a small owner-led boutique hotel, "
                "and the people who welcome you.",
            },
            "sections": [
                {
                    "type": "hero",
                    "title": "Our story",
                    "content_brief": "A warm opening that says who runs the hotel and why.",
                    "audience_needs": ["Clear photos"],
                },
                {
                    "type": "about",
                    "title": "How it began",
                    "content_brief": "A short story of how the hotel began, in a calm voice.",
                    "audience_needs": ["Easy booking"],
                },
            ],
        }
    }
    script = {
        "manager:EditIntent": [intent("add_page", "An About Us page")],
        "wireframe:PageOrder": [plan],
    }
    site = FakeSiteLLM(copies={**sample_site_copy(), "about": about})
    with api(edit_context(script, site)) as client:
        thread_id = finished_site(client)
        edit(client, thread_id, "Create an About Us page")
        assert page(client, thread_id, 2, "about.html").status_code == 200
        assert client.post(f"/runs/{thread_id}/versions/1/restore").status_code == 201
        assert page(client, thread_id, 3, "about.html").status_code == 404
        assert 'href="about.html"' not in page(client, thread_id, 3).text


def test_the_preview_serves_pages_with_a_policy_that_blocks_scripts_and_the_network() -> None:
    with api(edit_context({})) as client:
        thread_id = finished_site(client)
        home = page(client, thread_id, 1, "")
        css = page(client, thread_id, 1, "assets/base.css")

    assert home.status_code == 200
    assert home.headers["content-type"] == "text/html; charset=utf-8"
    assert css.headers["content-type"] == "text/css; charset=utf-8"
    policy = home.headers["content-security-policy"]
    assert "default-src 'none'" in policy
    assert "script-src" not in policy  # nothing allows a script: it falls to default-src
    assert "form-action 'none'" in policy
    assert "frame-ancestors http://localhost:3000" in policy
    assert home.headers["x-content-type-options"] == "nosniff"
    assert home.headers["cache-control"] == "no-cache"


def test_the_preview_answers_404_for_what_is_not_there() -> None:
    with api(edit_context({})) as client:
        thread_id = finished_site(client)
        assert page(client, thread_id, 1, "nothing.html").status_code == 404
        assert page(client, thread_id, 5).status_code == 404
        assert page(client, "missing", 1).status_code == 404
        # Files are looked up by their exact saved path, never on a disk, so no path escapes.
        assert page(client, thread_id, 1, "etc/passwd").status_code == 404
        assert page(client, thread_id, 0).status_code == 422
        error = page(client, thread_id, 5).json()["error"]
        assert error["code"] == "not_found"


def test_a_message_that_breaks_the_graph_leaves_the_run_finished() -> None:
    class Broken(FakeEditLLM):
        def structured(self, role: str, schema: Any) -> Any:
            if schema.__name__ == "EditIntent":
                msg = "the model is down"
                raise RuntimeError(msg)
            return super().structured(role, schema)

    context = edit_context({})
    context.llm = Broken({})
    with api(context) as client:
        thread_id = finished_site(client)
        events = edit(client, thread_id, "Change the hero")
        assert events[-1][0] in {"done", "error"}
        assert client.get(f"/runs/{thread_id}").json()["status"] == "completed"
        assert say(client, thread_id, "Try again").status_code == 202  # the lock is released
        finish(client, thread_id)
