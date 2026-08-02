"""``RunOutcome`` and ``OutcomeStore``: a run's terminal result, kept past the bus.

The per-run EventBus closes when a run ends, so the orchestrator would have no
way to observe an agent that finished while it was not watching. This store is
the durable substitute.

Split the same way as ``registry``: :class:`OutcomeRows` does the row work on an
OPEN ``Connection`` so the completion path can commit an outcome in the same
transaction as the agent row and the session record, and :class:`OutcomeStore`
wraps each call in one transaction for the callers that touch only outcomes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Connection, Row, delete, insert, select, update

from ..db import Database
from ..db.models import AgentOutcomeRow
from ..enums import ORCHESTRATOR_ID, AgentState

logger = logging.getLogger(__name__)

# The states that mean "this agent needs the orchestrator": a question it asked,
# a result it reported, a crash, or an approval sitting with the operator.
PENDING_STATES = (
    AgentState.WAITING,
    AgentState.REPORTED,
    AgentState.ERROR,
    AgentState.BLOCKED,
)


class RunOutcome(BaseModel):
    """The durable terminal outcome of an agent's most recent run: the final
    message + terminal state, persisted PAST the ephemeral per-run EventBus so
    the orchestrator can observe a finished agent later. ``acknowledged`` lets
    the orchestrator mark an outcome handled so it stops showing up as
    pending."""

    state: AgentState
    message: str = ""
    run_id: str = ""
    session_id: str | None = None
    ts: float = 0.0
    acknowledged: bool = False


def _record(row: Row[Any]) -> RunOutcome:
    """The pydantic record for one selected row. Nothing else leaves the store."""
    values = dict(row._mapping)
    values.pop("agent_id", None)
    return RunOutcome.model_validate(values)


class OutcomeRows:
    """The ``agent_outcome`` table, on a connection the CALLER opened."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def get(self, agent_id: str) -> RunOutcome | None:
        row = self._conn.execute(
            select(AgentOutcomeRow.__table__).where(
                AgentOutcomeRow.agent_id == agent_id
            )
        ).first()
        return None if row is None else _record(row)

    def all(self) -> dict[str, RunOutcome]:
        rows = self._conn.execute(select(AgentOutcomeRow.__table__)).all()
        return {row.agent_id: _record(row) for row in rows}

    def set(self, agent_id: str, outcome: RunOutcome) -> None:
        """Write the agent's outcome, replacing whatever was there.

        Delete-then-insert rather than a dialect-specific upsert: there is at
        most one row per agent, the whole record is being replaced, and both
        statements are inside the caller's transaction, so nothing can observe
        the gap between them.
        """
        self._conn.execute(
            delete(AgentOutcomeRow).where(AgentOutcomeRow.agent_id == agent_id)
        )
        self._conn.execute(
            insert(AgentOutcomeRow).values(
                agent_id=agent_id, **outcome.model_dump(mode="json")
            )
        )

    def clear(self, agent_id: str) -> None:
        self._conn.execute(
            delete(AgentOutcomeRow).where(AgentOutcomeRow.agent_id == agent_id)
        )

    def _signal(
        self,
        agent_id: str,
        state: AgentState,
        message: str,
        run_id: str,
        session_id: str | None,
    ) -> RunOutcome:
        """Write one unacknowledged mid-run signal outcome and return it. The
        caller has already checked that the agent exists."""
        outcome = RunOutcome(
            state=state,
            message=message,
            run_id=run_id,
            session_id=session_id,
            ts=time.time(),
            acknowledged=False,
        )
        self.set(agent_id, outcome)
        return outcome

    def awaiting_approval(
        self,
        agent_id: str,
        summary: str,
        *,
        run_id: str = "",
        session_id: str | None = None,
    ) -> RunOutcome:
        """Record that an agent has proposed a host action and is waiting for the
        OPERATOR to decide: a BLOCKED outcome carrying the rendered proposal.

        BLOCKED, not WAITING, and the difference is the DECIDER. A WAITING agent is
        one the orchestrator answers (`message_agent` resumes it with a reply); a
        BLOCKED one is waiting on an approval only a human with a session - or an
        allowlisted Telegram chat - can give. Routing an approval through WAITING
        would invite the orchestrator to answer "approved, go ahead", which it has
        no authority to say and which no code path would honour anyway.

        Keyed to the current ``run_id`` like the other signals, so the turn-end
        DONE preserves it (see ``AgentStore.mark_finished``)."""
        return self._signal(agent_id, AgentState.BLOCKED, summary, run_id, session_id)

    def request_input(
        self,
        agent_id: str,
        question: str,
        *,
        run_id: str = "",
        session_id: str | None = None,
    ) -> RunOutcome:
        """Record that a (mid-run) agent is blocked and needs a decision: write a
        WAITING outcome carrying ``question``, keyed to the current ``run_id`` so
        the turn-end DONE preserves it (see ``AgentStore.mark_finished``)."""
        return self._signal(agent_id, AgentState.WAITING, question, run_id, session_id)

    def report_back(
        self,
        agent_id: str,
        summary: str,
        *,
        run_id: str = "",
        session_id: str | None = None,
    ) -> RunOutcome:
        """Record that a (mid-run) agent has FINISHED its task and reported a
        result: write a REPORTED outcome carrying ``summary``, keyed to the current
        ``run_id`` so the turn-end DONE preserves it (see
        ``AgentStore.mark_finished``). The sibling of ``request_input`` for the
        completion case - the orchestrator reads the report and acknowledges it
        rather than resuming the agent."""
        return self._signal(agent_id, AgentState.REPORTED, summary, run_id, session_id)

    def pending(self) -> dict[str, RunOutcome]:
        """The agents with an UNACKNOWLEDGED signal: needs-input (`WAITING`),
        reported-done (`REPORTED`), `ERROR`, or awaiting an operator approval
        (`BLOCKED`). A cleanly DONE agent that did not report is not pending; an
        acknowledged one has been handled.

        BLOCKED is included so the orchestrator SEES that a delegated host change
        is sitting with the operator instead of concluding the agent went quiet -
        but it is a row to read, not one to answer: the message-an-agent path
        refuses a BLOCKED agent for an agent-credential caller, and `acknowledge`
        refuses to clear it, because the decision clears it when it lands.

        The reserved orchestrator is excluded: this list is the orchestrator's
        OWN "who needs me" poll, so it is never a member of it (mirrors
        `AgentStore.list()` hiding the orchestrator). Its own turns persist an
        ERROR outcome on a StreamError, which would otherwise make it
        self-appear."""
        rows = self._conn.execute(
            select(AgentOutcomeRow.__table__).where(
                AgentOutcomeRow.agent_id != ORCHESTRATOR_ID,
                AgentOutcomeRow.acknowledged.is_(False),
                AgentOutcomeRow.state.in_([s.value for s in PENDING_STATES]),
            )
        ).all()
        return {row.agent_id: _record(row) for row in rows}

    def acknowledge(self, agent_id: str) -> bool:
        """Mark an agent's outcome handled so it drops out of `pending`. Returns
        True if it flipped an unacknowledged outcome, False if there was none or it
        was already acknowledged (idempotent; never raises for an unknown agent - a
        deleted agent has no outcome to ack).

        The UPDATE carries its own ``acknowledged = 0`` predicate rather than
        reading the row and then writing it: ``rowcount`` is then the answer, and
        two concurrent acknowledgements cannot both report that they were the one
        that flipped it.

        Whether a BLOCKED outcome may be cleared is NOT decided here: this store
        knows nothing about proposals, and the answer depends on whether the
        approval is still live (see `host_approvals.live_for_agent`). The route
        enforces that policy, so an approval nobody decided cannot leave the agent
        with an outcome that can never be cleared."""
        result = self._conn.execute(
            update(AgentOutcomeRow)
            .where(
                AgentOutcomeRow.agent_id == agent_id,
                AgentOutcomeRow.acknowledged.is_(False),
            )
            .values(acknowledged=True)
        )
        return result.rowcount > 0


class OutcomeStore:
    """The persisted `(agent_id -> most-recent run outcome)` mapping - the
    durable substitute for the per-run EventBus, which closes when a run ends.

    One transaction per call. Not gated by ``settings_writable`` - like the
    run-state mutators it records server-internal run progress, not a user config
    edit."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, agent_id: str) -> RunOutcome | None:
        with self._db.transaction() as conn:
            return OutcomeRows(conn).get(agent_id)

    def all(self) -> dict[str, RunOutcome]:
        with self._db.transaction() as conn:
            return OutcomeRows(conn).all()

    def set(self, agent_id: str, outcome: RunOutcome) -> None:
        with self._db.transaction() as conn:
            OutcomeRows(conn).set(agent_id, outcome)

    def clear(self, agent_id: str) -> None:
        with self._db.transaction() as conn:
            OutcomeRows(conn).clear(agent_id)

    def awaiting_approval(
        self,
        agent_id: str,
        summary: str,
        *,
        run_id: str = "",
        session_id: str | None = None,
    ) -> RunOutcome:
        with self._db.transaction() as conn:
            return OutcomeRows(conn).awaiting_approval(
                agent_id, summary, run_id=run_id, session_id=session_id
            )

    def request_input(
        self,
        agent_id: str,
        question: str,
        *,
        run_id: str = "",
        session_id: str | None = None,
    ) -> RunOutcome:
        with self._db.transaction() as conn:
            return OutcomeRows(conn).request_input(
                agent_id, question, run_id=run_id, session_id=session_id
            )

    def report_back(
        self,
        agent_id: str,
        summary: str,
        *,
        run_id: str = "",
        session_id: str | None = None,
    ) -> RunOutcome:
        with self._db.transaction() as conn:
            return OutcomeRows(conn).report_back(
                agent_id, summary, run_id=run_id, session_id=session_id
            )

    def pending(self) -> dict[str, RunOutcome]:
        with self._db.transaction() as conn:
            return OutcomeRows(conn).pending()

    def acknowledge(self, agent_id: str) -> bool:
        with self._db.transaction() as conn:
            return OutcomeRows(conn).acknowledge(agent_id)
