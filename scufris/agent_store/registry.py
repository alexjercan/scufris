"""``SessionRegistry``: the only home of session ids and session ownership.

Two layers, and the split is what lets the completion path be ONE transaction:

- :class:`SessionRows` takes an OPEN ``Connection`` and does the row work. The
  agent store passes its own transaction down, so an agent's terminal state, its
  session record and its outcome commit together.
- :class:`SessionRegistry` owns a ``Database`` and wraps each call in one
  transaction, for the single-record callers that have no transaction of their
  own.

Nesting is an error on this engine (see ``scufris/db/engine.py``), so passing the
connection down is not an optimization - it is the only way one unit of work can
span two stores.

Session ids are backend-keyed: a codex rollout id means nothing to claude, so
every accessor matches on ``backend`` and a stale cross-backend id is
structurally unreachable. A backend switch starts a fresh history.

Not gated by ``settings_writable``: like the run-state mutators this records
server-internal run progress, not a user config edit.
"""

from __future__ import annotations

from sqlalchemy import Connection, delete, insert, select, update

from ..db import Database
from ..db.models import AgentSessionHistoryRow, AgentSessionRow

# The backend of a row that exists only to carry a spawn parent, recorded before
# the child has ever run. It matches no real backend, so every id accessor reads
# through it as "no mapping" until `_fresh` upgrades the row in place.
_NO_BACKEND = ""


class SessionRows:
    """The session tables, on a connection the CALLER opened."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def _row(self, agent_id: str, backend: str) -> tuple[str | None, bool]:
        """``(current_session_id, matched)`` for the agent UNDER ``backend``.

        ``matched`` is False both when there is no row and when the row belongs
        to another backend, which is what makes a cross-backend id unreachable
        everywhere rather than only in ``get``.
        """
        row = self._conn.execute(
            select(AgentSessionRow.backend, AgentSessionRow.current_session_id).where(
                AgentSessionRow.agent_id == agent_id
            )
        ).first()
        if row is None or row.backend != backend:
            return (None, False)
        return (row.current_session_id, True)

    def _fresh(self, agent_id: str, backend: str, session_id: str | None) -> None:
        """Restart the agent under ``backend`` with an empty history.

        The spawn parent is READ AND REWRITTEN rather than left alone: it is a
        fact about who spawned the child, independent of which backend it later
        runs under, so a backend switch must not drop it.
        """
        parent = self.parent_of(agent_id)
        self._conn.execute(
            delete(AgentSessionHistoryRow).where(
                AgentSessionHistoryRow.agent_id == agent_id
            )
        )
        self._conn.execute(
            delete(AgentSessionRow).where(AgentSessionRow.agent_id == agent_id)
        )
        self._conn.execute(
            insert(AgentSessionRow).values(
                agent_id=agent_id,
                backend=backend,
                current_session_id=session_id,
                parent_agent_id=parent[0],
                parent_session_id=parent[1],
            )
        )
        if session_id:
            self._append(agent_id, session_id)

    def _append(self, agent_id: str, session_id: str) -> None:
        """Add ``session_id`` to the end of the agent's history if it is new.

        The next ``seq`` is read inside the caller's transaction, which is what
        closes the read-modify-write window the JSON list had: every begin on
        this engine is immediate, so no other writer can take the sequence number
        between the read and the insert.
        """
        if session_id in self.sessions(agent_id):
            return
        highest = self._conn.execute(
            select(AgentSessionHistoryRow.seq)
            .where(AgentSessionHistoryRow.agent_id == agent_id)
            .order_by(AgentSessionHistoryRow.seq.desc())
            .limit(1)
        ).scalar()
        self._conn.execute(
            insert(AgentSessionHistoryRow).values(
                agent_id=agent_id,
                seq=0 if highest is None else highest + 1,
                session_id=session_id,
            )
        )

    def sessions(self, agent_id: str) -> list[str]:
        """The agent's history in minted order, backend-agnostic."""
        return list(
            self._conn.execute(
                select(AgentSessionHistoryRow.session_id)
                .where(AgentSessionHistoryRow.agent_id == agent_id)
                .order_by(AgentSessionHistoryRow.seq)
            ).scalars()
        )

    def get(self, agent_id: str, backend: str) -> str | None:
        """The agent's current session id under ``backend``, or None when there
        is no mapping or the stored ids belong to another backend."""
        return self._row(agent_id, backend)[0]

    def sessions_for(self, agent_id: str, backend: str) -> list[str]:
        """The agent's full session history under ``backend`` (the switcher list),
        or ``[]`` on a backend mismatch / no mapping."""
        if not self._row(agent_id, backend)[1]:
            return []
        return self.sessions(agent_id)

    def has(self, agent_id: str) -> bool:
        """Whether ANY mapping exists for this agent (backend-agnostic; the
        legacy-migration guard)."""
        return (
            self._conn.execute(
                select(AgentSessionRow.agent_id).where(
                    AgentSessionRow.agent_id == agent_id
                )
            ).first()
            is not None
        )

    def add(self, agent_id: str, backend: str, session_id: str) -> None:
        """Record a newly-minted session: set it current AND append it to the
        history (deduped). A backend change starts a fresh history."""
        if not self._row(agent_id, backend)[1]:
            self._fresh(agent_id, backend, session_id)
            return
        self._conn.execute(
            update(AgentSessionRow)
            .where(AgentSessionRow.agent_id == agent_id)
            .values(current_session_id=session_id)
        )
        self._append(agent_id, session_id)

    def set_current(self, agent_id: str, backend: str, session_id: str | None) -> None:
        """Switch to (or, with None, clear) the current session WITHOUT dropping
        history - this is "new chat" / "switch chat". A switched-to id not yet in
        the history is appended; a backend change starts a fresh history."""
        if not self._row(agent_id, backend)[1]:
            self._fresh(agent_id, backend, session_id)
            return
        self._conn.execute(
            update(AgentSessionRow)
            .where(AgentSessionRow.agent_id == agent_id)
            .values(current_session_id=session_id)
        )
        if session_id:
            self._append(agent_id, session_id)

    def remove(self, agent_id: str, backend: str, session_id: str) -> None:
        """Drop one session from the agent's history (a session delete), clearing
        ``current`` if it was that id. No-op on a backend mismatch / unknown id.

        The read of the current pointer and the two writes are the caller's ONE
        transaction, so a concurrent ``add`` cannot land between them and leave
        the deleted id current.
        """
        current, matched = self._row(agent_id, backend)
        if not matched:
            return
        self._conn.execute(
            delete(AgentSessionHistoryRow).where(
                AgentSessionHistoryRow.agent_id == agent_id,
                AgentSessionHistoryRow.session_id == session_id,
            )
        )
        if current == session_id:
            self._conn.execute(
                update(AgentSessionRow)
                .where(AgentSessionRow.agent_id == agent_id)
                .values(current_session_id=None)
            )

    def clear(self, agent_id: str) -> None:
        """Forget the agent's sessions entirely - its history, its current
        pointer and its spawn parent."""
        self._conn.execute(
            delete(AgentSessionHistoryRow).where(
                AgentSessionHistoryRow.agent_id == agent_id
            )
        )
        self._conn.execute(
            delete(AgentSessionRow).where(AgentSessionRow.agent_id == agent_id)
        )

    def set_parent(
        self,
        agent_id: str,
        parent_agent_id: str | None,
        parent_session_id: str | None,
    ) -> None:
        """Record which agent + session spawned ``agent_id``.

        Parent is a backend-independent fact, so this works even when the child
        has no session row yet: a placeholder is created under ``_NO_BACKEND``
        and later upgraded in place by ``_fresh`` when the child actually runs.
        """
        if self.has(agent_id):
            self._conn.execute(
                update(AgentSessionRow)
                .where(AgentSessionRow.agent_id == agent_id)
                .values(
                    parent_agent_id=parent_agent_id,
                    parent_session_id=parent_session_id,
                )
            )
            return
        self._conn.execute(
            insert(AgentSessionRow).values(
                agent_id=agent_id,
                backend=_NO_BACKEND,
                current_session_id=None,
                parent_agent_id=parent_agent_id,
                parent_session_id=parent_session_id,
            )
        )

    def parent_of(self, agent_id: str) -> tuple[str | None, str | None]:
        """The ``(parent_agent_id, parent_session_id)`` recorded for ``agent_id``,
        or ``(None, None)``. Backend-agnostic (parent is not session-specific)."""
        row = self._conn.execute(
            select(
                AgentSessionRow.parent_agent_id, AgentSessionRow.parent_session_id
            ).where(AgentSessionRow.agent_id == agent_id)
        ).first()
        if row is None:
            return (None, None)
        return (row.parent_agent_id, row.parent_session_id)


class SessionRegistry:
    """The persisted `(agent_id -> backend session history)` mapping - the ONLY
    home of session ids AND session ownership, for ALL agents (the orchestrator
    included).

    One transaction per call, for callers that touch only sessions. A caller that
    is already inside a unit of work uses :class:`SessionRows` on its connection
    instead; re-entering ``transaction()`` raises.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, agent_id: str, backend: str) -> str | None:
        with self._db.transaction() as conn:
            return SessionRows(conn).get(agent_id, backend)

    def sessions_for(self, agent_id: str, backend: str) -> list[str]:
        with self._db.transaction() as conn:
            return SessionRows(conn).sessions_for(agent_id, backend)

    def has(self, agent_id: str) -> bool:
        with self._db.transaction() as conn:
            return SessionRows(conn).has(agent_id)

    def set(self, agent_id: str, backend: str, session_id: str) -> None:
        """Back-compat alias of ``add`` (append a minted session + re-current)."""
        self.add(agent_id, backend, session_id)

    def add(self, agent_id: str, backend: str, session_id: str) -> None:
        with self._db.transaction() as conn:
            SessionRows(conn).add(agent_id, backend, session_id)

    def set_current(self, agent_id: str, backend: str, session_id: str | None) -> None:
        with self._db.transaction() as conn:
            SessionRows(conn).set_current(agent_id, backend, session_id)

    def remove(self, agent_id: str, backend: str, session_id: str) -> None:
        with self._db.transaction() as conn:
            SessionRows(conn).remove(agent_id, backend, session_id)

    def clear(self, agent_id: str) -> None:
        with self._db.transaction() as conn:
            SessionRows(conn).clear(agent_id)

    def set_parent(
        self,
        agent_id: str,
        parent_agent_id: str | None,
        parent_session_id: str | None,
    ) -> None:
        with self._db.transaction() as conn:
            SessionRows(conn).set_parent(agent_id, parent_agent_id, parent_session_id)

    def parent_of(self, agent_id: str) -> tuple[str | None, str | None]:
        with self._db.transaction() as conn:
            return SessionRows(conn).parent_of(agent_id)
