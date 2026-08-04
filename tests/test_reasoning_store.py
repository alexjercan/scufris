"""Tests for the reasoning sidecar store.

The sidecar is ``reasoning_turn`` rows now, so each test pairs the `database`
fixture with the store built on it. The store captures codex "thinking" per
(session, turn) so a hard reload can re-show the spoiler (reasoning is not on
disk).

The tolerant-load tests this file used to carry are gone on purpose: a damaged
per-session JSON file was the shape that swallowed 186 of 200 turns in
20260729-102146, and there is no longer a file to damage. What replaces them is
the proof BELOW that a failed append raises.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

from scufris.db import Database
from scufris.db.models import ReasoningTurnRow
from scufris.reasoning_store import ReasoningStore
from scufris.sessions import reasoning_fingerprint


def test_append_then_read_roundtrips_in_order(database: Database) -> None:
    store = ReasoningStore(database)
    store.append("sess-1", "first thought", answer="answer one")
    store.append("sess-1", "second thought", answer="answer two")

    entries = store.read("sess-1")
    assert [e.reasoning for e in entries] == ["first thought", "second thought"]
    # The answer is stored as its alignment fingerprint, not the raw text.
    assert entries[0].answer == reasoning_fingerprint("answer one")


def test_sessions_are_isolated(database: Database) -> None:
    store = ReasoningStore(database)
    store.append("sess-a", "a-think", answer="a")
    store.append("sess-b", "b-think", answer="b")

    assert [e.reasoning for e in store.read("sess-a")] == ["a-think"]
    assert [e.reasoning for e in store.read("sess-b")] == ["b-think"]


def test_empty_reasoning_still_records_an_entry(database: Database) -> None:
    # A turn with no thinking still gets an entry (empty), to keep the sidecar
    # 1:1 with the assistant messages the transcript surfaces.
    store = ReasoningStore(database)
    store.append("sess-1", "", answer="just an answer")
    entries = store.read("sess-1")
    assert len(entries) == 1
    assert entries[0].reasoning == ""


def test_read_unknown_session_is_empty(database: Database) -> None:
    store = ReasoningStore(database)
    assert store.read("never-seen") == []
    assert store.read(None) == []


def test_unsafe_session_id_is_a_noop(database: Database) -> None:
    store = ReasoningStore(database)
    # An id that does not look like a session id is a no-op, not a stored row
    # nothing will ever read back.
    store.append("../escape", "evil", answer="x")
    store.append("a/b", "evil", answer="x")
    assert store.read("../escape") == []
    with database.transaction() as conn:
        assert conn.execute(ReasoningTurnRow.__table__.select()).all() == []


def test_reasoning_turns_persist_without_swallowing_errors(database: Database) -> None:
    """A turn whose reasoning cannot be recorded says so.

    The file-backed store swallowed every write error here, which is why 186 of
    200 turns disappeared in 20260729-102146 with no failed request anywhere.
    Dropping the table stands in for the I/O failure the OSError swallow was
    about: what matters is that the caller SEES the failure instead of getting a
    silent no-op and a log line nobody reads.
    """
    store = ReasoningStore(database)
    store.append("sess-1", "recorded", answer="a")
    with database.transaction() as conn:
        conn.execute(text("DROP TABLE reasoning_turn"))

    with pytest.raises(DatabaseError):
        store.append("sess-1", "lost", answer="b")
