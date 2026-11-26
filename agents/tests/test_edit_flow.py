"""A chat message on a finished site, through the real graph: what is saved, what stays the same."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from test_research import text_of
from webkiln_agents.manager.context import Context, run_config
from webkiln_agents.manager.editing import DIGEST_CHARS, DIGEST_EDITS, edit_command, edit_digest
from webkiln_agents.manager.graph import build_graph
from webkiln_agents.manager.state import State
from webkiln_agents.memory.embeddings import EMBEDDING_DIMS
from webkiln_agents.memory.service import Memory
from webkiln_agents.site.checks import check_site
from webkiln_agents.site.copy import PageCopy
from webkiln_agents.site.edits import EditIntent
from webkiln_agents.testing import (
    APPROVE,
    SAMPLE_BRIEF,
    FakeEditLLM,
    FakeEmbeddings,
    FakeSiteLLM,
    sample_copy,
    sample_site_copy,
    sample_wireframe,
    site_context,
)
from webkiln_agents.workers.editing import DesignChange, PageOrder

WIREFRAME = sample_wireframe()
HERO = "index-hero"


def intent(kind: str, instruction: str = "Do the thing the user asked.", **fields: Any) -> dict:
    return {"intent": kind, "confidence": 0.9, "instruction": instruction, **fields}


def new_words(section_id: str, kind: str, heading: str) -> dict:
    """Valid copy for one section, from the shared sample copy."""
    return sample_copy([(section_id, kind, heading)])["sections"][0]


class Session:
    """A built site and the graph it lives in. `ask` sends one chat message to it."""

    def __init__(self, llm: FakeEditLLM, memory: Memory | None = None) -> None:
        self.llm = llm
        self.context: Context = site_context(llm)
        self.context.memory = memory
        self.graph = build_graph(InMemorySaver())
        self.config = run_config(self.context, "t1")
        self.events: list[Any] = []

    async def build(self) -> Session:
        await self.graph.ainvoke(dict(SAMPLE_BRIEF), self.config, context=self.context)
        await self.graph.ainvoke(Command(resume=APPROVE), self.config, context=self.context)
        return self

    async def ask(self, text: str, base_version: int | None = None) -> State:
        async for mode, chunk in self.graph.astream(
            edit_command(text, base_version),
            self.config,
            context=self.context,
            stream_mode=["custom"],
        ):
            assert mode == "custom"
            self.events.append(chunk)
        return await self.state()

    async def state(self) -> State:
        return State.model_validate((await self.graph.aget_state(self.config)).values)


async def built(script: dict[str, list[Any]] | None = None, **kw: Any) -> Session:
    return await Session(FakeEditLLM(script, **kw)).build()


def files_of(state: State) -> dict[str, str]:
    assert state.site is not None
    return state.site["files"]


def build_files(state: State) -> dict[str, str]:
    return state.outputs["generate"]["files"]


def last_reply(state: State) -> str:
    assert isinstance(state.messages[-1], AIMessage)
    return str(state.messages[-1].content)


async def test_a_content_edit_changes_one_section_and_only_its_page_file() -> None:
    words = new_words(HERO, "hero", "Wake up to the forest")
    session = await built(
        {
            "manager:EditIntent": [intent("content_edit", "Warmer hero heading", section=HERO)],
            "content:SectionCopy": [words],
        }
    )
    state = await session.ask("Make the homepage hero heading warmer", base_version=1)

    before, after = build_files(state), files_of(state)
    assert [p for p in before if before[p] != after[p]] == ["index.html"]  # all else identical
    assert "Wake up to the forest" in after["index.html"]
    assert set(after) == set(before)
    assert state.site is not None
    assert state.site["version"] == 2
    [change] = state.site_changes
    assert (change["version"], change["parent_version"], change["cause"]) == (2, 1, "content_edit")
    assert set(change["changed"]) == {"index.html"}
    assert change["sections"] == [HERO]
    assert check_site(after, WIREFRAME) == []


async def test_the_other_sections_of_the_page_keep_their_words() -> None:
    session = await built(
        {
            "manager:EditIntent": [intent("content_edit", section=HERO)],
            "content:SectionCopy": [new_words(HERO, "hero", "Wake up to the forest")],
        }
    )
    state = await session.ask("Change the hero")
    assert state.site is not None
    old = PageCopy.model_validate(state.outputs["generate"]["copies"]["index"])
    new = PageCopy.model_validate(state.site["copies"]["index"])
    assert new.sections[1:] == old.sections[1:]
    assert new.sections[0].heading == "Wake up to the forest"
    assert state.site["copies"]["rooms"] == state.outputs["generate"]["copies"]["rooms"]


async def test_the_reply_and_the_events_tell_the_chat_and_the_preview() -> None:
    session = await built(
        {
            "manager:EditIntent": [intent("content_edit", section=HERO)],
            "content:SectionCopy": [new_words(HERO, "hero", "Wake up to the forest")],
        }
    )
    state = await session.ask("Change the hero")
    assert "version 2" in last_reply(state)
    assert isinstance(state.messages[-2], HumanMessage)
    updated = next(e for e in session.events if e["event"] == "site_updated")
    assert updated["version"] == 2
    assert updated["paths"] == ["index.html"]
    assert updated["sections"] == [HERO]
    assert any(e["event"] == "edit_reply" for e in session.events)
    assert state.edit_request is None


async def test_a_design_edit_rewrites_the_tokens_and_writes_no_page_again() -> None:
    session = await built(
        {
            "manager:EditIntent": [intent("design_edit", "Darker, more luxurious scheme")],
            "brand:DesignChange": [
                {"primary": "#0b3d2e", "background": "#f4efe6", "rationale": "A deeper green."}
            ],
        }
    )
    calls = dict(session.llm.site.calls)
    state = await session.ask("Make it darker and more luxurious")

    before, after = build_files(state), files_of(state)
    assert [p for p in before if before[p] != after[p]] == ["assets/tokens.css"]
    assert "--color-primary: #0b3d2e;" in after["assets/tokens.css"]
    assert dict(session.llm.site.calls) == calls  # no page and no picture was written again
    assert state.site is not None
    assert state.site["brand"]["palette"]["primary"] == "#0b3d2e"
    assert state.site["brand"]["palette"]["secondary"] == "#c9a24b"  # the rest is kept
    assert state.site_changes[0]["cause"] == "design_edit"


async def test_a_new_font_changes_the_pages_that_link_it_and_nothing_else() -> None:
    session = await built(
        {
            "manager:EditIntent": [intent("design_edit")],
            "brand:DesignChange": [{"heading_font": "Lora", "rationale": "A softer serif."}],
        }
    )
    state = await session.ask("Use Lora for headings")
    before, after = build_files(state), files_of(state)
    changed = {p for p in before if before[p] != after[p]}
    assert changed == {"assets/tokens.css", *(p for p in before if p.endswith(".html"))}
    assert "family=Lora" in after["index.html"]
    assert after["assets/base.css"] == before["assets/base.css"]


async def test_a_palette_that_cannot_be_read_is_sent_back_to_the_model() -> None:
    session = await built(
        {
            "manager:EditIntent": [intent("design_edit")],
            "brand:DesignChange": [
                {"primary": "#f5f0e8", "rationale": "Nearly the background."},  # no contrast
                {"primary": "#0b3d2e", "rationale": "A deep green."},
            ],
        }
    )
    state = await session.ask("Make the main colour very light")
    assert state.site is not None
    assert state.site["version"] == 2
    retry = text_of(session.llm.prompts_for("brand", DesignChange)[1])
    assert "contrast" in retry
    assert state.site["brand"]["palette"]["primary"] == "#0b3d2e"


async def test_adding_a_section_changes_only_that_page() -> None:
    session = await built(
        {
            "manager:EditIntent": [
                intent(
                    "add_section",
                    "A testimonials section about quiet nights",
                    page="rooms",
                    section_type="testimonials",
                )
            ],
            "content:SectionCopy": [
                new_words("rooms-testimonials", "testimonials", "Quiet nights")
            ],
        }
    )
    state = await session.ask("Add a testimonials section to the rooms page")
    before, after = build_files(state), files_of(state)
    assert [p for p in before if before[p] != after[p]] == ["rooms.html"]
    assert 'id="rooms-testimonials"' in after["rooms.html"]
    assert state.site is not None
    ids = [s["id"] for s in state.site["wireframe"]["pages"][1]["sections"]]
    assert ids == ["rooms-hero", "rooms-gallery", "rooms-faq", "rooms-testimonials"]
    assert state.site_changes[0]["sections"] == ["rooms-testimonials"]


async def test_a_new_section_goes_before_a_closing_call_to_action() -> None:
    session = await built(
        {
            "manager:EditIntent": [
                intent("add_section", page="index", section_type="faq", instruction="A FAQ")
            ],
            "content:SectionCopy": [new_words("index-faq", "faq", "Good to know")],
        }
    )
    state = await session.ask("Add an FAQ to the homepage")
    assert state.site is not None
    ids = [s["id"] for s in state.site["wireframe"]["pages"][0]["sections"]]
    assert ids[-2:] == ["index-faq", "index-cta-band"]
    assert [s["id"] for s in state.site["copies"]["index"]["sections"]] == ids


async def test_removing_a_section_writes_no_words() -> None:
    session = await built(
        {"manager:EditIntent": [intent("remove_section", section="index-testimonials")]}
    )
    calls = dict(session.llm.site.calls)
    state = await session.ask("Remove the testimonials from the homepage")
    before, after = build_files(state), files_of(state)
    assert [p for p in before if before[p] != after[p]] == ["index.html"]
    assert "index-testimonials" not in after["index.html"]
    assert dict(session.llm.site.calls) == calls
    assert session.llm.prompts_for("content") == []


async def test_adding_a_page_writes_it_and_links_it_in_the_menu_before_contact() -> None:
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
    about = sample_copy(
        [("about-hero", "hero", "Our story"), ("about-about", "about", "How it began")],
        "about boutique hotel",
    )
    site = FakeSiteLLM(copies={**sample_site_copy(), "about": about})
    session = await built(
        {
            "manager:EditIntent": [intent("add_page", "An About Us page")],
            "wireframe:PageOrder": [plan],
        },
        site=site,
    )
    state = await session.ask("Create an About Us page")

    assert state.site is not None
    assert "about.html" in files_of(state)
    assert state.site["wireframe"]["navigation"] == ["index", "rooms", "offers", "about", "contact"]
    assert 'href="about.html"' in files_of(state)["index.html"]  # every page's menu changed
    assert PageOrder.model_validate(plan).page.slug == "about"
    assert state.site_changes[0]["sections"] == ["about-hero", "about-about"]
    assert check_site(files_of(state), session_wireframe(state)) == []


def session_wireframe(state: State):
    from webkiln_agents.workers.wireframe import Wireframe

    assert state.site is not None
    return Wireframe.model_validate(state.site["wireframe"])


async def test_a_question_is_answered_and_makes_no_version() -> None:
    session = await built(
        {
            "manager:EditIntent": [
                intent(
                    "question",
                    "What fonts?",
                    reply="Playfair Display for headings, Inter for text.",
                )
            ]
        }
    )
    state = await session.ask("What fonts are we using?")
    assert last_reply(state) == "Playfair Display for headings, Inter for text."
    assert state.site is None
    assert state.site_changes == []
    assert session.llm.prompts_for("content") == []


async def test_an_unsure_reading_asks_instead_of_guessing() -> None:
    session = await built(
        {
            "manager:EditIntent": [
                {
                    **intent("content_edit", section=HERO),
                    "confidence": 0.3,
                    "reply": "Do you mean the homepage hero or the rooms hero?",
                }
            ]
        }
    )
    state = await session.ask("Change the hero")
    assert last_reply(state) == "Do you mean the homepage hero or the rooms hero?"
    assert state.site is None
    assert session.llm.prompts_for("content") == []


async def test_a_section_that_does_not_exist_gets_a_question_with_the_real_ids() -> None:
    session = await built({"manager:EditIntent": [intent("content_edit", section="index-nothing")]})
    state = await session.ask("Change the thing")
    assert "index-hero" in last_reply(state)
    assert state.site is None


async def test_an_edit_made_against_an_old_version_is_refused_before_any_model_call() -> None:
    session = await built(
        {
            "manager:EditIntent": [intent("content_edit", section=HERO)],
            "content:SectionCopy": [new_words(HERO, "hero", "Wake up to the forest")],
        }
    )
    await session.ask("Change the hero", base_version=1)
    calls = len(session.llm.prompts_for("manager"))
    state = await session.ask("Change it again", base_version=1)  # the head is version 2 now
    assert "version 2" in last_reply(state)
    assert len(session.llm.prompts_for("manager")) == calls
    assert state.site is not None
    assert state.site["version"] == 2


async def test_a_section_that_never_passes_its_checks_is_refused_and_nothing_is_saved() -> None:
    wrong = new_words("index-features", "hero", "Not the right id")
    session = await built(
        {
            "manager:EditIntent": [intent("content_edit", section=HERO)],
            "content:SectionCopy": [wrong],
        }
    )
    state = await session.ask("Change the hero")
    assert "could not" in last_reply(state)
    assert state.site is None
    assert state.site_changes == []


async def test_a_model_that_keeps_failing_ends_as_a_chat_reply_and_the_run_stays_finished() -> None:
    session = await built(
        {
            "manager:EditIntent": [intent("content_edit", section=HERO)],
            "content:SectionCopy": [RuntimeError("the provider is down")],
        }
    )
    state = await session.ask("Change the hero")
    assert "nothing was saved" in last_reply(state)
    assert state.site is None
    assert state.site_changes == []
    assert (await session.graph.aget_state(session.config)).next == ()  # nothing left to resume


async def test_a_message_the_manager_cannot_read_leaves_the_site_alone() -> None:
    session = await built({"manager:EditIntent": [None]})
    state = await session.ask("???")
    assert "say it again" in last_reply(state)
    assert state.site is None


async def test_a_later_message_can_refer_to_an_earlier_instruction() -> None:
    session = await built(
        {
            "manager:EditIntent": [
                intent("content_edit", "Make the hero heading shorter", section=HERO),
                intent(
                    "content_edit",
                    "Make the rooms hero heading shorter, the same way as the homepage hero",
                    section="rooms-hero",
                ),
            ],
            "content:SectionCopy": [
                new_words(HERO, "hero", "Stay in the trees"),
                new_words("rooms-hero", "hero", "Sleep in the trees"),
            ],
        }
    )
    await session.ask("Make the homepage hero heading shorter", base_version=1)
    state = await session.ask("Do the same for the rooms page", base_version=2)

    second = text_of(session.llm.prompts_for("manager", EditIntent)[1])
    assert "Make the homepage hero heading shorter" in second  # the earlier instruction
    assert "version 2" in second  # and the reply to it
    assert second.rindex("Do the same for the rooms page") > second.index("<message>")
    section_prompt = text_of(session.llm.prompts_for("content", None)[1])
    assert "the same way as the homepage hero" in section_prompt
    assert state.site is not None
    assert state.site["version"] == 3
    assert "Sleep in the trees" in files_of(state)["rooms.html"]
    assert "Stay in the trees" in files_of(state)["index.html"]  # the first edit is kept


async def test_a_restored_thread_goes_on_from_the_newest_version() -> None:
    session = await built(
        {
            "manager:EditIntent": [
                intent("content_edit", section=HERO),
                intent("content_edit", section="rooms-hero"),
            ],
            "content:SectionCopy": [
                new_words(HERO, "hero", "Stay in the trees"),
                new_words("rooms-hero", "hero", "Sleep in the trees"),
            ],
        }
    )
    await session.ask("First")
    state = await session.ask("Second")
    assert [c["version"] for c in state.site_changes] == [2, 3]
    assert [c["parent_version"] for c in state.site_changes] == [1, 2]
    assert set(state.site_changes[1]["changed"]) == {"rooms.html"}  # only what the second changed


def memory_session(llm: FakeEditLLM) -> tuple[Session, InMemoryStore]:
    store = InMemoryStore(
        index={"dims": EMBEDDING_DIMS, "embed": FakeEmbeddings(), "fields": ["text"]}
    )
    return Session(llm, Memory(store, "u1", "p1")), store


async def test_a_lasting_preference_and_the_decision_are_remembered_and_recalled() -> None:
    llm = FakeEditLLM(
        {
            "manager:EditIntent": [
                intent(
                    "content_edit",
                    "A shorter hero heading",
                    section=HERO,
                    preferences=["Always keep headings under five words"],
                ),
                intent("content_edit", "Shorten the rooms hero too", section="rooms-hero"),
            ],
            "content:SectionCopy": [
                new_words(HERO, "hero", "Stay in the trees"),
                new_words("rooms-hero", "hero", "Sleep in the trees"),
            ],
        }
    )
    session, store = memory_session(llm)
    await session.build()
    await session.ask("Shorten the hero heading, and always keep headings under five words")
    texts = [i.value["text"] for i in await store.asearch(("user", "u1"), limit=20)]
    assert any("keep headings under five words" in t for t in texts)  # the preference
    assert any("Rewrote the hero section" in t for t in texts)  # the decision

    await session.ask("Shorten the rooms hero too")
    second = text_of(llm.prompts_for("manager", EditIntent)[1])
    assert "<memory>\n" in second
    assert "keep headings under five words" in second
    assert any(e["event"] == "memory_saved" for e in session.events)


def block_of(prompt_text: str, name: str) -> str:
    """The text between `<name>` and `</name>` in one prompt."""
    start = prompt_text.index(f"<{name}>")
    return prompt_text[start : prompt_text.index(f"</{name}>", start)]


async def test_the_router_reads_every_earlier_edit_even_after_the_chat_window_has_moved_on() -> (
    None
):
    wanted = [f"Change the hero heading, wording number {n}" for n in range(1, 5)]
    session = await built(
        {
            "manager:EditIntent": [
                *[intent("content_edit", text, section=HERO) for text in wanted],
                intent("question", "Why?", reply="Because."),
            ],
            "content:SectionCopy": [new_words(HERO, "hero", f"Heading {n}") for n in range(1, 5)],
        }
    )
    for n in range(1, len(wanted) + 1):
        await session.ask(f"Edit {n}", base_version=n)
    await session.ask("What did I change first?")

    last = str(session.llm.prompts_for("manager", EditIntent)[4][1].content)  # the user part
    chat, edits = block_of(last, "chat"), block_of(last, "edits")
    assert "Edit 1" not in chat  # four edits are eight messages: the first has left the window
    for n, text in enumerate(wanted, start=1):
        assert f"version {n + 1} (content_edit): {text}" in edits  # but the digest has all four
    assert edits.index("number 1") < edits.index("number 4")  # oldest first


async def test_the_first_message_reads_an_empty_digest_and_a_long_one_keeps_the_newest() -> None:
    session = await built(
        {"manager:EditIntent": [intent("question", "Hi", reply="Hello.")]},
    )
    await session.ask("Hello?")
    first = str(session.llm.prompts_for("manager", EditIntent)[0][1].content)
    assert "(no edits yet)" in block_of(first, "edits")

    state = await session.state()
    long_changes = [
        {"version": n + 2, "cause": "content_edit", "request": "x" * 400, "summary": "s"}
        for n in range(DIGEST_EDITS + 5)
    ]
    state = state.model_copy(update={"site_changes": long_changes})
    lines = edit_digest(state)
    assert len(lines) == DIGEST_EDITS
    assert lines[0].startswith("version 7 ")  # the five oldest are dropped
    assert all(len(line) < DIGEST_CHARS + 40 for line in lines)  # and each line is clipped
