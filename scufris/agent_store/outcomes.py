"""``RunOutcome`` and ``OutcomeStore``: a run's terminal result, kept past the bus.

The per-run EventBus closes when a run ends, so the orchestrator would have no
way to observe an agent that finished while it was not watching. This store is
the durable substitute.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from pydantic import BaseModel

from ..config import Settings
from ..enums import ORCHESTRATOR_ID, AgentState

logger = logging.getLogger(__name__)


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


class OutcomeStore:
    """The persisted `(agent_id -> most-recent run outcome)` mapping - the
    durable substitute for the per-run EventBus, which closes when a run ends.
    Mirrors ``SessionRegistry``: one JSON file under the state dir, atomic write,
    tolerant load. Not gated by ``settings_writable`` - like the run-state
    mutators it records server-internal run progress, not a user config edit."""

    def __init__(self, settings: Settings) -> None:
        self._path = Path(settings.state_dir) / "outcomes.json"
        self._outcomes: dict[str, RunOutcome] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("outcome store: cannot read %s: %s", self._path, exc)
            return
        if not isinstance(data, dict):
            return
        for agent_id, entry in data.items():
            if not (isinstance(agent_id, str) and isinstance(entry, dict)):
                continue
            try:
                self._outcomes[agent_id] = RunOutcome.model_validate(entry)
            except ValueError as exc:
                logger.warning("outcome store: dropping invalid outcome: %s", exc)

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {aid: o.model_dump() for aid, o in self._outcomes.items()}
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, self._path)

    def get(self, agent_id: str) -> RunOutcome | None:
        return self._outcomes.get(agent_id)

    def all(self) -> dict[str, RunOutcome]:
        return dict(self._outcomes)

    def set(self, agent_id: str, outcome: RunOutcome) -> None:
        self._outcomes[agent_id] = outcome
        self._persist()

    def clear(self, agent_id: str) -> None:
        if self._outcomes.pop(agent_id, None) is not None:
            self._persist()

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
        return {
            agent_id: outcome
            for agent_id, outcome in self._outcomes.items()
            if agent_id != ORCHESTRATOR_ID
            and not outcome.acknowledged
            and outcome.state
            in (
                AgentState.WAITING,
                AgentState.REPORTED,
                AgentState.ERROR,
                AgentState.BLOCKED,
            )
        }

    def acknowledge(self, agent_id: str) -> bool:
        """Mark an agent's outcome handled so it drops out of `pending`. Returns
        True if it flipped an unacknowledged outcome, False if there was none or it
        was already acknowledged (idempotent; never raises for an unknown agent - a
        deleted agent has no outcome to ack).

        Whether a BLOCKED outcome may be cleared is NOT decided here: this store
        knows nothing about proposals, and the answer depends on whether the
        approval is still live (see `host_approvals.live_for_agent`). The route
        enforces that policy, so an approval nobody decided cannot leave the agent
        with an outcome that can never be cleared."""
        outcome = self._outcomes.get(agent_id)
        if outcome is None or outcome.acknowledged:
            return False
        self.set(agent_id, outcome.model_copy(update={"acknowledged": True}))
        return True
