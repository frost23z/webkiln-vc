"""Chat about the finished site, and its versions."""

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path, Request
from langchain_core.messages import AIMessage, HumanMessage
from webkiln_agents.site.head import SiteChange, SiteHead, current_head

from webkiln_api import store
from webkiln_api.models import (
    ChatEntry,
    MessageRequest,
    RestoreResponse,
    RunResponse,
    VersionInfo,
    thread_config,
)
from webkiln_api.openapi import ERROR_RESPONSES
from webkiln_api.runs import get_stored_run, supervisor_of
from webkiln_api.site import RESTORE_CAUSE

router = APIRouter(prefix="/runs", tags=["site"], responses=ERROR_RESPONSES)

BUSY = "The site is being changed right now. Wait for it to finish, then send your message again."
NOT_BUILT = "The site is not built yet, so there is nothing to change."


def refusal(status: str) -> str:
    """Why a run cannot take a chat message or a restore, written for the user."""
    return {
        "running": BUSY,
        "waiting": "The brand guideline is waiting for your decision. Approve it first.",
        "failed": "The run failed before the site was finished. Retry it first.",
    }.get(status, BUSY)


async def claim(request: Request, thread_id: str) -> None:
    """Take the run lock, or refuse clearly. A run is `running` while it edits."""
    sessions = request.app.state.sessions
    run = await get_stored_run(request, thread_id)
    if supervisor_of(request).is_running(thread_id) or run.status != "completed":
        raise HTTPException(status_code=409, detail=refusal(run.status))
    if await store.latest_version(sessions, thread_id) is None:
        raise HTTPException(status_code=409, detail=NOT_BUILT)
    if not await store.claim_edit(sessions, thread_id):  # another request took the lock first
        raise HTTPException(status_code=409, detail=BUSY)


@router.post("/{thread_id}/messages", status_code=202)
async def send_message(thread_id: str, body: MessageRequest, request: Request) -> RunResponse:
    """A chat message about the finished site: change a section, the colours or the structure, or
    ask a question. Follow the answer with the stream route, and read the chat with GET.

    The run is `running` while it handles the message, so a second message on the same run gets
    409. `base_version` is the version the user was looking at: if the site has a newer one, the
    message is refused with 409 and nothing changes.
    """
    sessions = request.app.state.sessions
    latest = await store.latest_version(sessions, thread_id)
    if body.base_version is not None and latest is not None and body.base_version != latest:
        msg = (
            f"The site changed since you loaded it: you saw version {body.base_version} and it "
            f"is now version {latest}. Reload the preview and ask again."
        )
        raise HTTPException(status_code=409, detail=msg)
    await claim(request, thread_id)
    try:
        supervisor_of(request).send_message(thread_id, body.message, body.base_version)
    except Exception:
        await store.set_status(sessions, thread_id, "completed")  # nothing started
        raise
    return RunResponse(thread_id=thread_id, status="running")


@router.get("/{thread_id}/messages")
async def get_chat(thread_id: str, request: Request) -> list[ChatEntry]:
    """The chat about the site, oldest first."""
    await get_stored_run(request, thread_id)
    history = await store.chat_history(request.app.state.sessions, thread_id)
    return [ChatEntry.model_validate({"role": role, "content": text}) for role, text in history]


@router.get("/{thread_id}/versions")
async def list_versions(thread_id: str, request: Request) -> list[VersionInfo]:
    """The saved versions of the site, newest first."""
    await get_stored_run(request, thread_id)
    versions = await store.list_versions(request.app.state.sessions, thread_id)
    return [
        VersionInfo(
            version=v.version,
            parent_version=v.parent_version,
            cause=v.cause,
            summary=v.summary,
            created_at=v.created_at.isoformat(),
        )
        for v in versions
    ]


@router.post("/{thread_id}/versions/{version}/restore", status_code=201)
async def restore_version(
    thread_id: str, version: Annotated[int, Path(ge=1)], request: Request
) -> RestoreResponse:
    """Make a new version that is a copy of an older one. History is kept: nothing is deleted,
    and the restored version stays where it was."""
    sessions = request.app.state.sessions
    inputs = await store.site_version_inputs(sessions, thread_id, version)
    if inputs is None:
        raise HTTPException(status_code=404, detail="Site version not found")
    await claim(request, thread_id)
    try:
        new_version = await _restore(request, thread_id, version, inputs)
    except BaseException:
        await store.set_status(sessions, thread_id, "completed")  # nothing was changed
        raise
    return RestoreResponse(version=new_version, restored_from=version)


async def _restore(request: Request, thread_id: str, version: int, inputs: dict[str, Any]) -> int:
    """Write the restored version into the graph state, save it, and return its number."""
    sessions = request.app.state.sessions
    graph = request.app.state.graph
    config = thread_config(thread_id)
    values = (await graph.aget_state(config)).values
    head = current_head(values.get("site"), values.get("outputs", {}))
    if head is None:
        raise HTTPException(status_code=409, detail=NOT_BUILT)
    if version == head.version:
        raise HTTPException(status_code=409, detail="That is already the current version.")
    files = await store.site_files(sessions, thread_id, version)
    brand = inputs["brand"] or values.get("outputs", {}).get("brand")
    if not files or brand is None:
        raise HTTPException(status_code=409, detail="That version cannot be restored.")
    rendered = {
        "brand": brand,
        "wireframe": inputs["wireframe"],
        "copies": inputs["copies"],
        "images": inputs["images"],
    }
    change = SiteChange(
        version=head.version + 1,
        parent_version=head.version,
        cause=RESTORE_CAUSE,
        summary=f"Restored version {version}.",
        request=f"Restore version {version}",
        changed=files,  # a restore holds every file, see `store.site_files`
        sections=[],
        **rendered,
    )
    new_head = SiteHead(version=change.version, files=files, **rendered)
    update = {
        "site": new_head.model_dump(mode="json"),
        "site_changes": [change.model_dump(mode="json")],
        "messages": [
            HumanMessage(content=f"Go back to version {version}."),
            AIMessage(
                content=f"Restored version {version} as version {change.version}. "
                "Nothing was deleted."
            ),
        ],
    }
    # Written as `remember_edit`, so the run is at its end, as after any edit.
    await graph.aupdate_state(config, update, as_node="remember_edit")
    await request.app.state.supervisor.sync(thread_id, "completed")
    return change.version
