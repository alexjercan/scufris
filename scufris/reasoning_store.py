"""Sidecar store for codex reasoning ("thinking") text.

Codex persists reasoning only as an ``encrypted_content`` blob in its rollout,
so the plaintext is unrecoverable from disk. To let a hard page reload re-show
the thinking spoiler that streamed live, scufris captures the live
``reasoning_delta`` stream per (session, turn) into this sidecar, and
``CodexBackend.read_transcript`` merges it back into the transcript.

One ``reasoning_turn`` ROW per completed assistant turn, keyed
``(session_id, seq)`` and ordered oldest->newest. An entry is written even when
the model did not think (empty ``reasoning``), which is what keeps the list 1:1
with the assistant messages ``read_transcript`` surfaces. ``answer`` is a
whitespace-normalized fingerprint of the turn's final answer, used at merge time
to guard sidecar<->transcript alignment (see ``sessions.merge_reasoning``).

Rows rather than a JSON file per session (20260801-100409 DECISION.md 5): an
append is one insert instead of a rewrite whose cost grows with the session's
history.

**A failed append RAISES.** The file-backed store swallowed every ``OSError``
here on the theory that a lost sidecar entry degrades to "no reasoning on
reload" rather than a failed turn. What it actually bought was 186 of 200 turns
disappearing with no failed request anywhere (20260729-102146): the reasoning
the operator watched stream was simply not there afterwards, and nothing said
so. A turn that cannot record its reasoning is a turn that did not fully
succeed, and the caller is entitled to know.

It records server-internal turn output, not a user config edit, so it is not
gated by ``settings_writable``.
"""

from __future__ import annotations

import re

from pydantic import BaseModel
from sqlalchemy import insert, select

from .db import Database
from .db.models import ReasoningTurnRow
from .sessions import reasoning_fingerprint

# codex session ids are UUID-ish. The charset guard predates the database and is
# kept: it is now about what may become a PRIMARY KEY value rather than what may
# become a filename, but a session id that does not look like one is still a
# caller mistake, and answering it with a no-op is still better than storing a
# row nothing will ever read.
_SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9._-]+")


class ReasoningEntry(BaseModel):
    """One completed assistant turn's captured reasoning + its alignment key."""

    answer: str  # fingerprint of the turn's final answer text
    reasoning: str


def _usable(session_id: str | None) -> bool:
    """Whether this id can key a turn (missing or malformed -> a no-op)."""
    return bool(session_id) and _SAFE_SESSION_ID.fullmatch(session_id or "") is not None


class ReasoningStore:
    """Per-session captured reasoning text, in the state database."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def append(self, session_id: str | None, reasoning: str, *, answer: str) -> None:
        """Record one completed turn's reasoning, keyed by ``session_id``.

        ``answer`` is the turn's final answer text; it is fingerprinted for
        alignment. A missing or malformed id is a no-op; anything else that goes
        wrong raises, so a turn whose reasoning did not land says so.

        The next ``seq`` is read inside the same transaction as the insert, so
        two turns of one session completing together cannot claim the same
        sequence number.
        """
        if not _usable(session_id):
            return
        with self._db.transaction() as conn:
            highest = conn.execute(
                select(ReasoningTurnRow.seq)
                .where(ReasoningTurnRow.session_id == session_id)
                .order_by(ReasoningTurnRow.seq.desc())
                .limit(1)
            ).scalar()
            conn.execute(
                insert(ReasoningTurnRow).values(
                    session_id=session_id,
                    seq=0 if highest is None else highest + 1,
                    answer=reasoning_fingerprint(answer),
                    reasoning=reasoning,
                )
            )

    def read(self, session_id: str | None) -> list[ReasoningEntry]:
        """The session's captured turns, oldest-first; empty when absent."""
        if not _usable(session_id):
            return []
        with self._db.transaction() as conn:
            rows = conn.execute(
                select(ReasoningTurnRow.answer, ReasoningTurnRow.reasoning)
                .where(ReasoningTurnRow.session_id == session_id)
                .order_by(ReasoningTurnRow.seq)
            ).all()
        return [
            ReasoningEntry(answer=row.answer, reasoning=row.reasoning) for row in rows
        ]
