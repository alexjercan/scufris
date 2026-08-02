"""``AgentStore``'s signal half: what an agent says about itself mid-run.

A signal is an outcome written while the agent is still running - it proposed a
host action, it needs a decision, or it has finished and is reporting. All three
write an ``agent_outcome`` row and none of them touch the ``agents`` row, which
is why they are the seam this file cuts on.

Each opens ONE transaction and does the existence check inside it: the check and
the write are then atomic, so a delete that races a live sub-agent's callback
either loses (the outcome is written for an agent that still exists) or wins (the
callback raises ``AgentNotFound`` and writes nothing). Split around the write, a
deleted agent's outcome could be resurrected - the
completion-callback-write-after-existence-check lesson.

It is a MIXIN because ``AgentStore`` has one public surface and is at the
600-line source cap; it is not an extension point, and nothing else inherits it.
"""

from __future__ import annotations

from ..db import Database
from .outcomes import OutcomeRows, RunOutcome
from .rows import require_exists


class AgentSignals:
    """The mid-run signal mutators and the outcome reads over them."""

    _db: Database

    def awaiting_approval(
        self,
        agent_id: str,
        summary: str,
        *,
        run_id: str = "",
        session_id: str | None = None,
    ) -> RunOutcome:
        """Record that an agent has proposed a host action and is waiting for the
        OPERATOR to decide (see ``OutcomeRows.awaiting_approval``)."""
        with self._db.transaction() as conn:
            require_exists(conn, agent_id)
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
        """Record that a (mid-run) agent needs a decision (see
        ``OutcomeRows.request_input``). Raises AgentNotFound for a missing agent
        (the caller is a live sub-agent, but a delete could race), writing nothing
        in that case."""
        with self._db.transaction() as conn:
            require_exists(conn, agent_id)
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
        """Record that a (mid-run) agent has finished and reported a result (see
        ``OutcomeRows.report_back``). Raises AgentNotFound for a missing agent
        (the caller is a live sub-agent, but a delete could race), writing nothing
        in that case."""
        with self._db.transaction() as conn:
            require_exists(conn, agent_id)
            return OutcomeRows(conn).report_back(
                agent_id, summary, run_id=run_id, session_id=session_id
            )

    def outcome(self, agent_id: str) -> RunOutcome | None:
        """The agent's most-recent durable run outcome, or None if it has not
        finished a run yet."""
        with self._db.transaction() as conn:
            return OutcomeRows(conn).get(agent_id)

    def outcomes(self) -> dict[str, RunOutcome]:
        """All agents' most-recent run outcomes, keyed by agent id. The
        orchestrator's "who needs me" poll reads from here."""
        with self._db.transaction() as conn:
            return OutcomeRows(conn).all()

    def pending_outcomes(self) -> dict[str, RunOutcome]:
        """The agents with an UNACKNOWLEDGED signal (see
        ``OutcomeRows.pending``)."""
        with self._db.transaction() as conn:
            return OutcomeRows(conn).pending()

    def acknowledge(self, agent_id: str) -> bool:
        """Mark an agent's outcome handled so it drops out of `pending_outcomes`
        (see ``OutcomeRows.acknowledge``)."""
        with self._db.transaction() as conn:
            return OutcomeRows(conn).acknowledge(agent_id)
