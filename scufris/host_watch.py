"""One pass of the scheduled host checks.

The one thing in this app that starts without a person. `HostScheduler` owns the
clock; this owns what a run DOES: read the checks off the loop, render a digest,
deliver it (or not, per the schedule and the mute), and escalate a breach into
the ordinary approval queue if the operator has switched that on.

`muted` and `telegram_bot` are read through callables rather than held, and both
for the same reason: the scheduler is built AROUND this service (it takes
`run` as its callback), and the bot starts after the app is assembled. A held
reference would be the wrong one or no reference at all.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from scufris_host import HostInspector
from scufris_hostctl import (
    HostApprovalService,
    HostdClient,
    HostdError,
    HostdUnavailable,
)
from scufris_hostd import Requester

from .agent_diagnostics import AgentDiagnostics
from .agent_store import ORCHESTRATOR_ID, AgentStore
from .checks import CheckRun, run_checks
from .config import Settings
from .digest import DigestStore, render_digest
from .health import AgentHealth
from .scheduler import DAILY

logger = logging.getLogger(__name__)


# Who a threshold-driven proposal is recorded as. Not an agent and not the operator:
# the audit should say plainly that nobody asked for this - a check did.
SCHEDULED_CHECK_ACTOR = "scheduled-check"


class HostWatchService:
    """What one scheduled check pass does, end to end."""

    def __init__(
        self,
        *,
        settings: Settings,
        inspector: HostInspector,
        agents: AgentStore,
        diagnostics: AgentDiagnostics,
        digests: DigestStore,
        approvals: HostApprovalService,
        hostd: HostdClient,
        muted: Callable[[], bool],
        telegram_bot: Callable[[], Any | None],
    ) -> None:
        self._settings = settings
        self._inspector = inspector
        self._agents = agents
        self._diagnostics = diagnostics
        self._digests = digests
        self._approvals = approvals
        self._hostd = hostd
        self._muted = muted
        self._telegram_bot = telegram_bot

    async def run(self, schedule: str) -> str:
        """One pass of the checks for ``schedule``; returns the sentence to record."""
        if not self._settings.host_checks_enabled:
            return "skipped: host checks are disabled"

        async def health() -> AgentHealth:
            # `agents.get`, not `require_agent_async`: this runs off the HTTP
            # path, where raising an HTTPException would be wrong.
            orchestrator = await asyncio.to_thread(self._agents.get, ORCHESTRATOR_ID)
            return await self._diagnostics.health(orchestrator)

        previous = await asyncio.to_thread(self._digests.last_states)
        run = await run_checks(self._inspector, self._settings, health=health)
        digest = render_digest(
            run,
            previous=previous,
            schedule=schedule,
            # The daily schedule always speaks; `watch` only when something changed.
            always=schedule == DAILY,
        )
        # Escalate BEFORE reporting the outcome, so a proposal the digest mentions is
        # already in the queue when the operator reads it.
        escalated = await self._escalate_breaches(run, previous)
        if digest is None:
            return "ran: nothing to report" + (f"; {escalated}" if escalated else "")
        # Rebound: `add` assigns the row id, and `mark_delivered` keys on it.
        digest = await asyncio.to_thread(self._digests.add, digest)
        if self._muted():
            await asyncio.to_thread(self._digests.mark_delivered, digest, error="muted")
            return "ran and recorded; delivery muted" + (
                f"; {escalated}" if escalated else ""
            )
        error = await self._deliver_digest(digest.text)
        await asyncio.to_thread(self._digests.mark_delivered, digest, error=error)
        outcome = f"delivery failed: {error}" if error else "delivered"
        return f"ran ({digest.verdict}), {outcome}" + (
            f"; {escalated}" if escalated else ""
        )

    async def _deliver_digest(self, text: str) -> str:
        """Send the digest to the operator. Returns "" or why it could not.

        A delivery failure is not allowed to lose the digest: it is already in the
        store and readable on the /host/ page, and the schedule records that the
        message did not land. Being told late beats not being told and not knowing it.
        """
        bot = self._telegram_bot()
        if bot is None:
            return "no telegram bot is configured"
        try:
            return await bot.send_digest(text)
        except Exception as exc:  # noqa: BLE001 - a transport failure is a record
            logger.warning("digest delivery failed: %s", exc)
            return f"{type(exc).__name__}: {exc}"

    async def _escalate_breaches(self, run: CheckRun, previous: dict[str, str]) -> str:
        """Propose what a breached check asked for, if anything.

        The proposal goes through the ordinary approval service, so it is previewed,
        queued, announced and decided exactly like one an agent asked for - and it is
        never applied here. A check may only ask for what `checks.ESCALATABLE`
        allows, which `escalation_for` enforces at construction.

        TWO guards against asking repeatedly, and they are the difference between a
        helpful proposal and a queue full of identical ones (review round 1, R1.2):

        - only a check whose state CHANGED into the breach escalates. A store that has
          been full since yesterday has already asked;
        - and never while an equivalent proposal from these checks is still decidable.
          One pending collection is the ask; a second is noise.
        """
        proposed: list[str] = []
        pending_kinds = {
            record.proposal.kind
            for record in await self._approvals.decidable()
            if record.proposal.requester.actor == SCHEDULED_CHECK_ACTOR
        }
        for result in run.results:
            escalation = result.escalation
            if escalation is None:
                continue
            if previous.get(result.name) == result.state.value:
                logger.debug(
                    "not re-escalating %s: unchanged since the last digest", result.name
                )
                continue
            if escalation.kind in pending_kinds:
                logger.info(
                    "not escalating %s: a %s proposal is already waiting",
                    result.name,
                    escalation.kind,
                )
                continue
            try:
                proposal = await self._hostd.propose(
                    escalation.kind,
                    dict(escalation.args),
                    Requester(actor=SCHEDULED_CHECK_ACTOR, agent=result.name),
                )
            except (HostdUnavailable, HostdError) as exc:
                logger.info("could not escalate the %s check: %s", result.name, exc)
                continue
            await self._approvals.record_proposal(proposal)
            proposed.append(f"proposed {escalation.kind} ({proposal.id[:8]})")
        return ", ".join(proposed)


__all__ = ["SCHEDULED_CHECK_ACTOR", "HostWatchService"]
