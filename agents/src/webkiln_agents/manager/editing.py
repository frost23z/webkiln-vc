"""Conversational editing: the nodes that handle a chat message on the finished site.

The API sends a message as `Command(update={"messages": [...], "edit_request": {...}},
goto="edit_router")` on the finished thread. The graph has no entry branch at START, because the
first run always starts from the brief. The nodes:

    edit_router    one Manager call reads the message (`EditIntent`) with the last turns of the
                   chat and the user's remembered preferences, and code checks it against the site
    edit_content   one section gets new words          (one content call)
    edit_design    colours and fonts change             (one brand call, no page is written again)
    edit_structure add or remove a section, add a page
    remember_edit  the decision goes to long-term memory

Every edit renders the site again from its inputs with code (`rebuild`), so only the files whose
inputs changed differ, and runs the site checks before anything is saved. A refused edit, a
question and an unclear message end as a reply in the chat and make no version. The user never
sees a failure as a broken run: when a node's retries are used up, `edit_failed` replies instead.

The lock against two edits at once and the base-version check are in the API (a run is `running`
while it edits), and the router checks the base version again from the state.
"""

import logging
from collections.abc import Awaitable
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.errors import NodeError
from langgraph.runtime import Runtime
from langgraph.types import Command

from webkiln_agents.budget import CAP_REPLY, CallCapReached, cap_reached
from webkiln_agents.manager.context import Context
from webkiln_agents.manager.remember import Candidate, announce, clip, save, site_candidates
from webkiln_agents.manager.state import State
from webkiln_agents.memory.prompt import memory_block
from webkiln_agents.memory.records import Kind
from webkiln_agents.site.copy import SectionCopy
from webkiln_agents.site.edits import (
    EditIntent,
    Target,
    find_page,
    listing,
    next_section_id,
    page_from_plan,
    resolve,
    wireframe_problems,
    with_page,
    with_replaced_copy,
    with_section,
    without_section,
)
from webkiln_agents.site.head import EditRefused, SiteChange, SiteHead, current_head, rebuild
from webkiln_agents.workers.content import write_page
from webkiln_agents.workers.editing import ask_intent, design_change, plan_page, write_section
from webkiln_agents.workers.errors import WorkerOutputError
from webkiln_agents.workers.wireframe import Section

logger = logging.getLogger(__name__)

# The nodes of an edit. A run whose pending step is one of them is handling a chat message.
EDIT_NODES = frozenset(
    {"edit_router", "edit_content", "edit_design", "edit_structure", "remember_edit"}
)
HISTORY_TURNS = 6  # earlier messages the Manager reads, so "the same as before" can be resolved
DIGEST_EDITS = 30  # earlier edits the Manager reads, one line each
DIGEST_CHARS = 160  # longest line of that digest
RECALLED = 3  # remembered lines given to the Manager
Next = Literal["edit_content", "edit_design", "edit_structure", "__end__"]
# Where an edit node goes next. One type for all of them, because `Command` is invariant: a
# helper that returns `Command[Literal["__end__"]]` does not fit a node that can also save.
Done = Literal["remember_edit", "__end__"]


def edit_command(message: str, base_version: int | None = None) -> Command[Any]:
    """What the API sends for a chat message: the message, the version the user was looking at,
    and the node to enter. It goes to the thread the run finished on."""
    request = {"base_version": base_version} if base_version is not None else None
    update = {"messages": [HumanMessage(content=message)], "edit_request": request}
    return Command(update=update, goto="edit_router")


def _text(message: AnyMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else " ".join(str(part) for part in content)


def _latest_request(state: State) -> str | None:
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            return _text(message).strip() or None
    return None


def _history(state: State) -> list[tuple[str, str]]:
    """The turns before the latest message, oldest first, as `(role, text)`."""
    earlier = state.messages[:-1][-HISTORY_TURNS:]
    return [("user" if isinstance(m, HumanMessage) else "assistant", _text(m)) for m in earlier]


def edit_digest(state: State) -> list[str]:
    """One line for each edit saved so far, from `site_changes`, so the Manager still knows an
    instruction that is older than the chat window. The oldest lines go first when it is long."""
    lines = [
        f"version {c['version']} ({c['cause']}): {clip(c['request'] or c['summary'], DIGEST_CHARS)}"
        for c in state.site_changes
    ]
    return lines[-DIGEST_EDITS:]


def reply(text: str) -> dict[str, Any]:
    """A chat reply that changes nothing else. It also clears the request that was being handled."""
    get_stream_writer()({"event": "edit_reply", "message": text})
    return {"messages": [AIMessage(content=text)], "edit_request": None}


def _intent_of(state: State) -> EditIntent:
    return EditIntent.model_validate((state.edit_request or {})["intent"])


def _head(state: State) -> SiteHead:
    head = current_head(state.site, state.outputs)
    if head is None:
        msg = "The site is not built yet."
        raise EditRefused(msg)
    return head


def _target(state: State, head: SiteHead) -> Target:
    target = resolve(_intent_of(state), head.wireframe_model())
    if isinstance(target, str):  # the router already checked it, so this means the site moved
        raise EditRefused(target)
    return target


def _design_facts(state: State, head: SiteHead) -> dict[str, Any]:
    brand = head.brand
    return {
        "version": head.version,
        "palette": brand["palette"],
        "fonts": {k: brand["typography"][k] for k in ("heading_font", "body_font")},
        "tagline": brand["messaging"]["tagline"],
        "recent_changes": [c["summary"] for c in state.site_changes[-3:]],
    }


async def edit_router(state: State, runtime: Runtime[Context]) -> Command[Next]:
    """Read the message and send it to the node that handles it, or answer it right here."""
    message = _latest_request(state)
    head = current_head(state.site, state.outputs)
    if message is None:
        return Command(goto="__end__")
    if head is None:
        return Command(update=reply("The site is not built yet, so there is nothing to change."))
    seen = (state.edit_request or {}).get("base_version")
    if seen is not None and seen != head.version:
        text = (
            f"The site changed since you loaded it: you saw version {seen} and it is now "
            f"version {head.version}. Reload the preview and ask again."
        )
        return Command(update=reply(text), goto="__end__")
    context = runtime.context
    recalled = (
        await context.memory.recall(message, limit=RECALLED) if context and context.memory else []
    )
    intent = await ask_intent(
        context.llm if context else None,
        message,
        _history(state),
        listing(head.wireframe_model(), head.page_copies()),
        _design_facts(state, head),
        memory_block(recalled),
        edit_digest(state),
    )
    if intent is None:
        if cap_reached(context.llm if context else None):
            return Command(update=reply(CAP_REPLY), goto="__end__")
        text = "I could not work out what you want. Please say it again, a little differently."
        return Command(update=reply(text), goto="__end__")
    if context and context.memory and intent.preferences:
        announce("edit", await save(context.memory, _preferences(intent)))
    outcome = resolve(intent, head.wireframe_model())
    if isinstance(outcome, str):
        return Command(update=reply(outcome), goto="__end__")
    request = {"base_version": head.version, "intent": intent.model_dump(mode="json")}
    if outcome.intent == "content_edit":
        return Command(update={"edit_request": request}, goto="edit_content")
    if outcome.intent == "design_edit":
        return Command(update={"edit_request": request}, goto="edit_design")
    return Command(update={"edit_request": request}, goto="edit_structure")


def _preferences(intent: EditIntent) -> list[Candidate]:
    return [Candidate(clip(text), "preference", "user_said") for text in intent.preferences]


def _applied(head: SiteHead, change: SiteChange, text: str) -> Command[Done]:
    """The update for a saved edit: the new head, the change for the API, and the chat reply."""
    new_head = SiteHead(
        version=change.version,
        brand=change.brand,
        wireframe=change.wireframe,
        copies=change.copies,
        images=change.images,
        files={**head.files, **change.changed},
    )
    get_stream_writer()(
        {
            "event": "site_updated",
            "version": change.version,
            "parent_version": change.parent_version,
            "cause": change.cause,
            "summary": change.summary,
            "paths": sorted(change.changed),
            "sections": change.sections,
        }
    )
    update = reply(f"{text} The site is now at version {change.version}.")
    update["site"] = new_head.model_dump(mode="json")
    update["site_changes"] = [change.model_dump(mode="json")]
    return Command(update=update, goto="remember_edit")


def _refused(error: Exception) -> Command[Done]:
    text = str(error) if isinstance(error, EditRefused) else f"I could not do that: {error}"
    return Command(update=reply(text[:600]), goto="__end__")


def _brand_name(state: State) -> str:
    return state.initial_input.brand_name


async def _memory_text(runtime: Runtime[Context], query: str) -> str:
    context = runtime.context
    if context is None or context.memory is None:
        return ""
    return memory_block(await context.memory.recall(query, limit=RECALLED))


def _llm(runtime: Runtime[Context]):
    llm = runtime.context.llm if runtime.context else None
    if llm is None:
        msg = "No model is set up for edits, so nothing was changed."
        raise EditRefused(msg)
    return llm


# The router already checked the target, so these mean the site moved under the edit.
NO_SECTION = "I could not tell which section you mean."
NO_SECTION_TYPE = "I could not tell which section to add."
NO_PAGE = "I could not tell which page you mean."


async def _content_change(state: State, runtime: Runtime[Context]) -> tuple[SiteHead, SiteChange]:
    head = _head(state)
    target = _target(state, head)
    page, section = target.page, target.section
    if page is None or section is None:
        raise EditRefused(NO_SECTION)
    copies = head.page_copies()
    page_copy = copies[page.slug]
    current = next(s for s in page_copy.sections if s.id == section.id)
    words = await write_section(
        _llm(runtime),
        head.wireframe_model(),
        page,
        section,
        page_copy,
        current,
        target.instruction,
        head.brand,
        state.initial_input.model_dump(),
        await _memory_text(runtime, target.instruction),
    )
    copies[page.slug] = with_replaced_copy(page_copy, words)
    return rebuild(
        head,
        brand_name=_brand_name(state),
        cause="content_edit",
        summary=f"Rewrote the {section.type} section of the {page.name} page.",
        request=target.instruction,
        sections=[section.id],
        copies=copies,
    )


async def _design_change(state: State, runtime: Runtime[Context]) -> tuple[SiteHead, SiteChange]:
    head = _head(state)
    target = _target(state, head)
    memory = await _memory_text(runtime, target.instruction)
    guideline, values = await design_change(_llm(runtime), head.brand, target.instruction, memory)
    return rebuild(
        head,
        brand_name=_brand_name(state),
        cause="design_edit",
        summary=values.rationale.rstrip(".") + ".",
        request=target.instruction,
        brand=guideline.model_dump(),
    )


async def _structure_change(state: State, runtime: Runtime[Context]) -> tuple[SiteHead, SiteChange]:
    head = _head(state)
    target = _target(state, head)
    wireframe, copies = head.wireframe_model(), head.page_copies()
    brief = state.initial_input.model_dump()
    memory = await _memory_text(runtime, target.instruction)
    sections: list[str] = []
    page = target.page
    if target.intent == "remove_section":
        section = target.section
        if page is None or section is None:
            raise EditRefused(NO_SECTION)
        wireframe, copies[page.slug] = without_section(
            wireframe, copies[page.slug], page, section.id
        )
        summary = f"Removed the {section.type} section from the {page.name} page."
    elif target.intent == "add_section":
        if page is None or target.section_type is None:
            raise EditRefused(NO_SECTION_TYPE)
        fresh = Section(
            id=next_section_id(page, target.section_type),
            type=target.section_type,
            title=f"New {target.section_type.replace('_', ' ')} section",
            content_brief=clip(f"Added at the user's request: {target.instruction}", 400),
            audience_needs=["Asked for by the user in chat"],
        )
        placeholder = SectionCopy(id=fresh.id, heading=fresh.title)
        wireframe, copies[page.slug] = with_section(
            wireframe, copies[page.slug], page, fresh, placeholder
        )
        where = find_page(wireframe, page.slug)
        if where is None:
            raise EditRefused(NO_PAGE)
        words = await write_section(
            _llm(runtime), wireframe, where, fresh, copies[page.slug], None,
            target.instruction, head.brand, brief, memory,
        )  # fmt: skip
        copies[page.slug] = with_replaced_copy(copies[page.slug], words)
        sections = [fresh.id]
        summary = f"Added a {fresh.type} section to the {page.name} page."
    else:  # add_page
        audience = (state.outputs.get("audience") or {}).get("needs", [])
        order = await plan_page(
            _llm(runtime), wireframe, target.instruction, head.brand, brief, audience, memory
        )
        new_page = page_from_plan(order.page)
        wireframe = with_page(wireframe, new_page, order.functional_requirements)
        copies[new_page.slug] = await write_page(
            _llm(runtime),
            new_page,
            wireframe,
            head.brand,
            brief,
            state.analysis.model_dump() if state.analysis else None,
            target.instruction,
        )
        sections = [s.id for s in new_page.sections]
        summary = f"Added the {new_page.name} page."
    if problems := wireframe_problems(wireframe):
        msg = "That change breaks the site's structure: " + "; ".join(problems)
        raise EditRefused(msg[:600])
    return rebuild(
        head,
        brand_name=_brand_name(state),
        cause=target.intent,
        summary=summary,
        request=target.instruction,
        sections=sections,
        wireframe=wireframe,
        copies=copies,
    )


async def _finish(change_of: Awaitable[tuple[SiteHead, SiteChange]]) -> Command[Done]:
    """Save the edit, or answer in chat why it was refused. Other errors reach the retry policy."""
    try:
        head, change = await change_of
    except (EditRefused, WorkerOutputError) as error:
        return _refused(error)
    return _applied(head, change, change.summary)


async def edit_content(state: State, runtime: Runtime[Context]) -> Command[Done]:
    """Give one section new words. Every other section keeps its words, so only its page changes."""
    return await _finish(_content_change(state, runtime))


async def edit_design(state: State, runtime: Runtime[Context]) -> Command[Done]:
    """Change colours or fonts. `tokens.css` is written again, and the pages too when a font
    changes (their head links the font). No page is generated again."""
    return await _finish(_design_change(state, runtime))


async def edit_structure(state: State, runtime: Runtime[Context]) -> Command[Done]:
    """Add a section, remove a section or add a page."""
    return await _finish(_structure_change(state, runtime))


async def remember_edit(state: State, runtime: Runtime[Context]) -> None:
    """Keep the edit as a decision in long-term memory. A later one on the same subject replaces
    it (`Memory.remember` supersedes a close match), so memory holds the latest choice."""
    memory = runtime.context.memory if runtime.context else None
    if memory is None or not state.site_changes:
        return
    change = state.site_changes[-1]
    text = clip(f"Edit to the site: {change['summary']} The user asked: {change['request']}")
    candidates = [Candidate(text, "decision", "user_said")]
    # The site now differs from what memory says when the brand or the structure changed.
    try:
        candidates += site_candidates(
            _brand_name(state), change["cause"], change["brand"], change["wireframe"]
        )
    except ValueError:  # memory is best effort: the edit itself is already saved
        logger.exception("the site could not be read for memory")
    kinds: list[Kind] = await save(memory, candidates)
    announce("edit", kinds)


def edit_failed(state: State, error: NodeError) -> Command[Done]:
    """The error handler of the edit nodes, called once their retries are used up. The chat gets
    an apology and the run stays finished: the site is as it was before the message."""
    message = f"{type(error.error).__name__}: {error.error}"[:200]
    logger.warning("an edit failed after its retries: %s", message)
    if isinstance(error.error, CallCapReached):
        text = CAP_REPLY
    else:
        text = "I could not make that change just now, and nothing was saved. Please try again."
    return Command(update=reply(text), goto="__end__")
