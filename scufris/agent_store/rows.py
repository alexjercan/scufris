"""Reading the ``agents`` table: what every store method starts from.

These take an OPEN ``Connection`` because the store's units of work span more
than one table - the completion path writes an agent row, a session record and
an outcome in ONE transaction, and nesting is an error on this engine.

What crosses this boundary is always an :class:`AgentRecord`, never a row: the
FastAPI routes serialize what the store returns, and a row object in a response
is a detached-load failure waiting for the first attribute access.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Connection, Row, select

from ..db.models import AgentRow
from ..enums import HOST_AGENT_ID
from .records import AgentNotFound, AgentRecord
from .registry import SessionRows


def record(row: Row[Any]) -> AgentRecord:
    """The pydantic record for one selected row.

    ``session_id`` is absent from the table and defaults to None here: the
    session tables own it, and ``with_session`` attaches it at read time.
    """
    return AgentRecord.model_validate(dict(row._mapping))


def fetch(conn: Connection, agent_id: str) -> AgentRecord | None:
    row = conn.execute(
        select(AgentRow.__table__).where(AgentRow.id == agent_id)
    ).first()
    return None if row is None else record(row)


def require(conn: Connection, agent_id: str) -> AgentRecord:
    """The agent's stored record (session_id always None), or ``AgentNotFound``.

    Mutators build on this so a registry-attached id never leaks back into the
    row, and so a delete racing a live sub-agent's callback writes nothing.
    """
    agent = fetch(conn, agent_id)
    if agent is None:
        raise AgentNotFound(agent_id)
    return agent


def require_exists(conn: Connection, agent_id: str) -> None:
    """Existence guard for the signal mutators, which write an OUTCOME rather
    than an ``agents`` row.

    The HOST agent has no row and must still be able to signal: outcomes are
    keyed by agent id in their own table, so nothing here depends on the row
    existing. ``require`` is still the guard for a normal agent, so a delete
    racing a live sub-agent's callback writes nothing (the
    completion-callback-write-after-existence-check lesson).

    The ORCHESTRATOR is deliberately NOT exempt, even though it is equally
    synthetic: it registers no `agent` callback server, so it has no way to
    signal and no reason to - and a route that accepted its id would be
    accepting a caller that cannot exist.
    """
    if agent_id == HOST_AGENT_ID:
        return
    require(conn, agent_id)


def with_session(sessions: SessionRows, agent: AgentRecord) -> AgentRecord:
    """The record as the API sees it: its current session id attached from the
    session tables, keyed by the agent's CURRENT backend (a cross-backend id
    reads as None)."""
    return agent.model_copy(
        update={"session_id": sessions.get(agent.id, agent.backend)}
    )


def unique_id(conn: Connection, base: str) -> str:
    """``base``, or the first ``base-N`` free of it.

    Called INSIDE the caller's transaction, which is what closes the
    read-modify-write window: every begin on this engine is immediate, so no
    other writer - in this process or another - can take the id between the
    lookup here and the insert that follows.
    """
    taken = set(
        conn.scalars(
            # autoescape because a slug may contain '_', which is a LIKE
            # wildcard: unescaped, `a_b` would also match `axb` and inflate the
            # taken set.
            select(AgentRow.id).where(AgentRow.id.startswith(base, autoescape=True))
        ).all()
    )
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"
