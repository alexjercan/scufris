"""A pending host approval is a BLOCKED agent.

The requesting agent proposes and ends its turn; the operator decides; the
decision resumes the agent. That round trip runs on the machinery a sub-agent
already uses (the outcome store plus one launched turn), with ONE difference
that matters: the state is BLOCKED, not WAITING, because the decider is the
operator and not the orchestrator.

`connect` registers the bridge on the approval service. Draining a held decision
is exposed separately because it belongs to the RUN-completion fan-out, whose
order `create_app` owns.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from .agent_store import ORCHESTRATOR_ID, AgentNotFound, AgentRecord, AgentStore
from .host_actions import HostActionRecord
from .host_approvals import HostApprovalService, decision_message
from .orchestrator import AgentRunService, RunAlreadyActive
from .projects import ProjectNotFound, ProjectStore


class HostApprovalBridge:
    """Blocks the agent that asked, and resumes it with what the operator decided."""

    def __init__(
        self,
        *,
        agents: AgentStore,
        projects: ProjectStore,
        runs: AgentRunService,
        approvals: HostApprovalService,
        telegram_bot: Callable[[], Any | None],
    ) -> None:
        self._agents = agents
        self._projects = projects
        self._runs = runs
        self._approvals = approvals
        self._telegram_bot = telegram_bot
        # agent_id -> the decision text that could not be delivered because a turn
        # was in flight. Drained by the run-completion callback, like a deferred wake.
        self._deferred: dict[str, str] = {}
        self._notify_tasks: set[asyncio.Task[None]] = set()

    def connect(self) -> None:
        """Subscribe to the approval service's proposal and decision hooks."""
        self._approvals.on_proposed(self._mark_requester_blocked)
        self._approvals.on_proposed(
            lambda record: self._announce(record, decision=False)
        )
        self._approvals.on_decided(lambda record: self._announce(record, decision=True))
        # A proposal recovered from the helper after a restart marks its requester
        # too: that agent IS still waiting, and its persisted outcome should say so
        # rather than depending on which process wrote it.
        self._approvals.on_restored(self._mark_requester_blocked)
        self._approvals.on_decided(self._tell_requester_the_decision)

    async def _requesting_agent(self, record: HostActionRecord) -> AgentRecord | None:
        """The agent whose proposal this is, if an agent asked at all.

        The operator proposing from a surface has no agent to block, and the
        ORCHESTRATOR is never it: it holds no propose tool, and the identity helper
        labels a nameless machine caller "orchestrator" by default, so a proposal
        attributed to it means "some agent-credentialled caller that did not name
        itself" rather than a resumable agent turn.
        """
        agent_id = record.proposal.requester.agent.strip()
        if not agent_id or agent_id == ORCHESTRATOR_ID:
            return None
        try:
            return await asyncio.to_thread(self._agents.get, agent_id)
        except AgentNotFound:
            return None

    async def _mark_requester_blocked(self, record: HostActionRecord) -> None:
        """Record the requesting agent as BLOCKED on this proposal."""
        agent = await self._requesting_agent(record)
        if agent is None:
            return
        await self._runs.awaiting_approval(
            agent,
            f"waiting for the operator to decide host action {record.id}: "
            f"{record.proposal.summary}",
        )

    async def _deliver_decision(self, agent: AgentRecord, text: str) -> None:
        """Resume the agent with the decision, or hold it until its turn ends.

        `RunAlreadyActive` means a turn for that agent is already in flight (it
        proposed and kept working), and dropping the decision there would be the
        exact failure the denial path exists to prevent - so it is held and
        delivered by the completion callback instead. Only that refusal is held:
        an agent that has been deleted or whose project is gone is a real
        failure, not a race to retry.
        """
        try:
            project = (
                await asyncio.to_thread(self._projects.get, agent.project_id)
                if agent.project_id
                else None
            )
        except ProjectNotFound:
            project = None
        try:
            await self._runs.launch(agent, project, text)
        except RunAlreadyActive:
            held = self._deferred.get(agent.id)
            self._deferred[agent.id] = f"{held}\n\n{text}" if held else text

    async def _tell_requester_the_decision(self, record: HostActionRecord) -> None:
        """Hand a decided action's outcome back to the agent that asked for it."""
        agent = await self._requesting_agent(record)
        if agent is None:
            return
        text = decision_message(record)
        if text is None:
            return  # approved and still running: the result is the news
        await self._deliver_decision(agent, text)

    async def drain_deferred_decision(self, agent_id: str) -> None:
        """Deliver a decision that was held while the agent was mid-turn."""
        text = self._deferred.pop(agent_id, None)
        if text is None:
            return
        try:
            agent = await asyncio.to_thread(self._agents.get, agent_id)
        except AgentNotFound:
            return
        await self._deliver_decision(agent, text)

    def _announce(self, record: HostActionRecord, *, decision: bool) -> None:
        """Push a proposal, or a decision, into the operator's chat.

        Fire-and-forget on purpose: this is a NOTIFICATION, and a Telegram outage
        must not fail the decision that already happened or the proposal that is
        already in the queue. The hook layer logs whatever this raises.

        A restored proposal (recovered from the helper after a restart) deliberately
        does NOT come through here - see the on_restored wiring in `connect`:
        re-announcing old news on every restart is how a notification channel gets
        muted.
        """
        bot = self._telegram_bot()
        if bot is None:
            return
        coroutine = (
            bot.announce_decision(record) if decision else bot.announce_proposal(record)
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop: a store driven directly by a test or a CLI, where there is
            # no bot to notify anyway.
            coroutine.close()
            return
        task = loop.create_task(coroutine)
        # Held until it finishes, so the task is not garbage-collected mid-send.
        self._notify_tasks.add(task)
        task.add_done_callback(self._notify_tasks.discard)


__all__ = ["HostApprovalBridge"]
