"""The generated site inside the API: what is saved in Postgres and what the stream may carry.

The files of a site are large and are not part of a task's report. The graph state holds them (the
checkpoint is the truth), `site_files` holds the copy the API serves, and everything the clients
see, the stream and the run detail, gets the list of paths instead.
"""

from collections.abc import Mapping
from typing import Any

GENERATE_AGENT = "generate"
INITIAL_CAUSE = "generate"
RESTORE_CAUSE = "restore"  # a version that is a full copy of an older one


def is_site_output(data: Any) -> bool:
    """The output of `generate`: a dict with the files of the site."""
    return isinstance(data, dict) and isinstance(data.get("files"), dict) and "pages" in data


def without_files(data: Mapping[str, Any]) -> dict[str, Any]:
    """The output with its files replaced by their paths. Other outputs are returned as they are."""
    if not is_site_output(data):
        return dict(data)
    return {**{k: v for k, v in data.items() if k != "files"}, "file_paths": sorted(data["files"])}


def _slim_site(site: Any) -> Any:
    """The newest site (a `SiteHead`) with its files replaced by their paths."""
    if isinstance(site, dict) and isinstance(site.get("files"), dict):
        return {
            **{k: v for k, v in site.items() if k != "files"},
            "file_paths": sorted(site["files"]),
        }
    return site


def _slim_changes(changes: Any) -> Any:
    """The versions that edits made, with the changed files replaced by their paths."""
    if not isinstance(changes, list):
        return changes
    return [
        {**{k: v for k, v in c.items() if k != "changed"}, "paths": sorted(c["changed"])}
        if isinstance(c, dict) and isinstance(c.get("changed"), dict)
        else c
        for c in changes
    ]


def without_edit_files(values: Mapping[str, Any]) -> dict[str, Any]:
    """Graph state values with the files of the newest site and of each edit taken out."""
    slim = dict(values)
    if "site" in slim:
        slim["site"] = _slim_site(slim["site"])
    if "site_changes" in slim:
        slim["site_changes"] = _slim_changes(slim["site_changes"])
    return slim


def strip_update(data: Any) -> Any:
    """An `updates` event of the graph, without the files of a site.

    The event is `{node: update}`, and the update holds the outputs of the tasks that finished.
    """
    if not isinstance(data, dict):
        return data
    cleaned: dict[str, Any] = {}
    for node, update in data.items():
        outputs = update.get("outputs") if isinstance(update, dict) else None
        if isinstance(outputs, dict) and any(is_site_output(o) for o in outputs.values()):
            update = {**update, "outputs": {t: without_files(o) for t, o in outputs.items()}}
        if isinstance(update, dict) and ("site" in update or "site_changes" in update):
            update = without_edit_files(update)
        cleaned[node] = update
    return cleaned
