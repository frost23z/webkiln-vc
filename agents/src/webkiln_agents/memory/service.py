"""Long-term memory for one user and one project: write with dedupe and versions, and recall.

Writing
- Identical text is found by its hash and skipped, with no embedding call.
- A record in a `slot` (the approved brand, a research summary) replaces the current record of
  that slot. Anything else replaces the closest current record of its kind only when the two are
  very similar (`SUPERSEDE_ABOVE`), so a miss keeps both. Nothing is ever overwritten: the old
  record becomes `superseded`, and the new one links back to it with a higher version.

Reading
- Only `current` records are searched. The project is searched first, then the user's
  preferences, and a small recency boost is applied in code.
"""

import logging
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid7

from langgraph.store.base import BaseStore, SearchItem

from webkiln_agents.memory.records import (
    Kind,
    MemoryRecord,
    Recalled,
    Source,
    namespace_for,
    normalise,
    now,
    preferences_namespace,
    project_namespace,
    text_hash,
)

logger = logging.getLogger(__name__)

# Cosine similarity above which a new record of the same kind is taken to be an updated version of
# the old one. It is high on purpose: a miss only keeps both records. Tuned with `check-memory` on
# gemini-embedding-2: the same preference scored 0.886 or more and different ones 0.634 or less.
SUPERSEDE_ABOVE = 0.8
RECENCY_WEIGHT = 0.1  # at most +10% for a record written just now
RECENCY_HALF_LIFE_DAYS = 30.0

type Outcome = Literal["added", "duplicate", "superseded"]


def _record(item: SearchItem) -> MemoryRecord:
    return MemoryRecord.model_validate(item.value)


def _boost(score: float, created_at: str) -> float:
    age = (datetime.now(UTC) - datetime.fromisoformat(created_at)).total_seconds() / 86400
    return score * (1 + RECENCY_WEIGHT * math.pow(0.5, max(age, 0) / RECENCY_HALF_LIFE_DAYS))


class Memory:
    """The memory of one user in one project. Both ids go into every namespace."""

    def __init__(self, store: BaseStore, user_id: str, project_id: str) -> None:
        self.store = store
        self.user_id = user_id
        self.project_id = project_id

    async def remember(
        self, text: str, kind: Kind, source: Source, *, slot: str | None = None
    ) -> Outcome:
        """Save one sentence. Raises on a store or embedding failure: callers decide what a
        failure means (the graph nodes log it and carry on)."""
        text = normalise(text)
        digest = text_hash(text)
        ns = namespace_for(self.user_id, self.project_id, kind)
        if await self.store.asearch(ns, filter={"hash": digest, "status": "current"}, limit=1):
            return "duplicate"
        previous = await self._previous(ns, text, kind, slot)
        key = uuid7().hex
        record = MemoryRecord(
            text=text,
            kind=kind,
            source=source,
            slot=slot,
            version=_record(previous).version + 1 if previous else 1,
            supersedes=previous.key if previous else None,
            project_id=None if kind == "preference" else self.project_id,
            hash=digest,
            created_at=now(),
        )
        await self.store.aput(ns, key, record.model_dump())
        if previous:
            # The new record is saved first, so a stop between the two writes leaves two current
            # records, never none. `index=False` skips a new embedding for the old text.
            update = {"status": "superseded", "superseded_by": key}
            old = _record(previous).model_copy(update=update)
            await self.store.aput(ns, previous.key, old.model_dump(), index=False)
        return "superseded" if previous else "added"

    async def _previous(
        self, ns: tuple[str, ...], text: str, kind: Kind, slot: str | None
    ) -> SearchItem | None:
        if slot is not None:
            found = await self.store.asearch(
                ns, filter={"slot": slot, "status": "current"}, limit=1
            )
            return found[0] if found else None
        found = await self.store.asearch(
            ns, query=text, filter={"kind": kind, "status": "current"}, limit=1
        )
        if found and found[0].score is not None and found[0].score >= SUPERSEDE_ABOVE:
            return found[0]
        return None

    async def recall(
        self, query: str, *, kinds: Sequence[Kind] | None = None, limit: int = 3
    ) -> list[Recalled]:
        """The current records closest to `query`: up to `limit` from this project, then up to
        `limit` of the user's preferences. A failure is logged and gives what was found so far."""
        found: list[Recalled] = []
        for ns in (
            project_namespace(self.user_id, self.project_id),
            preferences_namespace(self.user_id),
        ):
            try:
                items = await self.store.asearch(
                    ns, query=query, filter={"status": "current"}, limit=limit * 3
                )
            except Exception:
                logger.exception("memory search failed")
                continue
            ranked = sorted(
                (
                    Recalled(
                        key=item.key,
                        score=_boost(item.score or 0.0, _record(item).created_at),
                        record=_record(item),
                    )
                    for item in items
                    if kinds is None or _record(item).kind in kinds
                ),
                key=lambda r: r.score,
                reverse=True,
            )
            found.extend(ranked[:limit])
        return found
