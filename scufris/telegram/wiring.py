"""What the bot is allowed to read and decide, and how it is started.

The transport is `bot.py`; this is the layer between it and the rest of the app.
Both provider bundles exist so the bot never talks HTTP to its own server: it
reads the SAME in-process services the web routes read, and it decides through
the SAME approval service.

Two rules live here rather than in the transport, and both are the rule the web
routes already follow:

- the ACTOR is derived, never supplied. The bot hands over a chat id; this builds
  `operator:telegram:<chat_id>`, so the audit says which surface decided and a
  transport cannot claim to be someone else (the web path derives its actor from
  the session cookie for the same reason);
- the allowlist is re-checked HERE. The bot already refuses a chat that is not
  allowlisted, and this refuses it again, so neither layer is the only thing
  between a stray chat and a root command.
"""

from __future__ import annotations

import asyncio
import logging

from scufris_host import Collector, HostStats

from ..agent_diagnostics import (
    AgentDiagnostics,
    mcp_servers_for_audience,
    tools_for_servers,
)
from ..agent_store import ORCHESTRATOR_ID, AgentStore
from ..config import Settings
from ..env_bridge import ensure_den_path
from ..health import AgentHealth
from ..host_actions import AlreadyDecided, HostActionRecord, UnknownAction
from ..host_approvals import (
    ConfirmationRequired,
    HostApprovalService,
    ProposalExpired,
)
from ..hostclient import HostdError, HostdUnavailable
from ..mcp_models import AgentTool
from ..orchestrator import OrchestratorTurnService
from .bot import TelegramBot
from .contracts import ApprovalOps, ApprovalOutcome, OrchestratorInfo, SettingsOps
from .orchestrator import build_telegram_callbacks

logger = logging.getLogger(__name__)


def build_approval_ops(
    settings: Settings, approvals: HostApprovalService
) -> ApprovalOps:
    """The bot's host-approval providers, wired to the ONE approval service.

    Every refusal message the operator reads is the service's own sentence
    ("already denied by ...", "this proposal has expired", "needs the explicit
    acknowledgement ..."), which is what keeps the two surfaces from developing
    different ideas of the same rule.
    """

    def _actor(chat_id: int) -> str:
        return f"operator:telegram:{chat_id}"

    def _refuse_unallowed(chat_id: int) -> ApprovalOutcome | None:
        if chat_id in set(settings.telegram_allowed_chat_ids):
            return None
        logger.warning(
            "refused a host decision from a non-allowlisted telegram chat %s",
            chat_id,
        )
        return ApprovalOutcome(ok=False, message="this chat cannot decide host actions")

    async def pending() -> list[HostActionRecord]:
        # Reconcile with the helper first, so a proposal made before a restart
        # (or by another client of the socket) is decidable from the phone too.
        try:
            await approvals.refresh_pending(
                min_interval=settings.host_queue_refresh_seconds
            )
        except (HostdUnavailable, HostdError) as exc:
            logger.debug("telegram queue reconcile skipped: %s", exc)
        # `decidable`, not "pending": a proposal whose window has closed, or
        # whose machine has drifted, must not come back with a button the
        # service would refuse.
        return await approvals.decidable()

    async def get(action_id: str) -> HostActionRecord | None:
        try:
            return await approvals.get(action_id)
        except UnknownAction:
            return None

    async def approve(
        action_id: str, chat_id: int, acknowledge: str
    ) -> ApprovalOutcome:
        refused = _refuse_unallowed(chat_id)
        if refused is not None:
            return refused
        try:
            record, _run_id = await approvals.approve(
                action_id, actor=_actor(chat_id), acknowledge=acknowledge
            )
        except UnknownAction:
            return ApprovalOutcome(ok=False, message="no such host action")
        except (
            ConfirmationRequired,
            AlreadyDecided,
            ProposalExpired,
            HostdUnavailable,
            HostdError,
        ) as exc:
            return ApprovalOutcome(ok=False, message=str(exc))
        return ApprovalOutcome(
            ok=True,
            message=(
                f"approved {record.proposal.summary} - applying it now; the "
                "result follows"
            ),
            record=record,
        )

    async def deny(action_id: str, chat_id: int, reason: str) -> ApprovalOutcome:
        refused = _refuse_unallowed(chat_id)
        if refused is not None:
            return refused
        # "-" is how the prompt offers "no reason", and an empty reason is
        # recorded as exactly that rather than as the literal dash.
        cleaned = "" if reason.strip() == "-" else reason.strip()
        try:
            record = await approvals.deny(
                action_id, actor=_actor(chat_id), reason=cleaned
            )
        except UnknownAction:
            return ApprovalOutcome(ok=False, message="no such host action")
        except (AlreadyDecided, HostdUnavailable, HostdError) as exc:
            return ApprovalOutcome(ok=False, message=str(exc))
        told = (
            " The agent that asked has been told why."
            if cleaned and record.proposal.requester.agent
            else ""
        )
        return ApprovalOutcome(
            ok=True,
            message=f"denied {record.proposal.summary}.{told}",
            record=record,
        )

    return ApprovalOps(pending=pending, get=get, approve=approve, deny=deny)


def build_settings_ops(
    settings: Settings,
    agents: AgentStore,
    diagnostics: AgentDiagnostics,
    collector: Collector,
) -> SettingsOps:
    """The read-only providers behind the bot's `/settings` and `/stats` commands,
    wired to the SAME in-process readers the web settings endpoints use (the
    diagnostics service, the orchestrator tool catalog, the host collector) -
    orchestrator-scoped, no self-HTTP."""

    async def info() -> OrchestratorInfo:
        def read() -> OrchestratorInfo:
            orchestrator = agents.get(ORCHESTRATOR_ID)
            # The account IS the service's single answer for auth mode, model,
            # enabled and quota, so `/settings` and `/settings usage` both come
            # from this one call instead of rebuilding those facts by hand.
            account = diagnostics.account(orchestrator)
            return OrchestratorInfo(
                backend=str(orchestrator.backend),
                model=account.model,
                auth_mode=(
                    str(account.auth_mode) if account.auth_mode is not None else None
                ),
                enabled=account.enabled,
                permission_mode=str(settings.agent_permission_mode),
                quota=account.quota,
            )

        # The WHOLE body off-loop: the store read opens a transaction, and the
        # codex quota reader rglobs + parses every rollout, so neither can stall
        # the bot's poll loop (R1.1).
        return await asyncio.to_thread(read)

    async def health() -> AgentHealth:
        orchestrator = await asyncio.to_thread(agents.get, ORCHESTRATOR_ID)
        ensure_den_path(settings)  # so the in-process den probe sees the den
        return await diagnostics.health(orchestrator)

    async def tools() -> list[AgentTool]:
        return await tools_for_servers(
            settings, mcp_servers_for_audience(ORCHESTRATOR_ID)
        )

    async def stats() -> HostStats:
        # collector.sample() is synchronous psutil I/O: off-loop (R1.1).
        return await asyncio.to_thread(collector.sample)

    return SettingsOps(info=info, health=health, tools=tools, stats=stats)


def start_bot(
    settings: Settings,
    turn: OrchestratorTurnService,
    *,
    settings_ops: SettingsOps,
    approval_ops: ApprovalOps,
) -> tuple[TelegramBot | None, "asyncio.Task[None] | None"]:
    """Launch the in-process Telegram bot when a token is configured.

    The bot drives the orchestrator through the SAME turn service as the landing
    chat via injected callbacks - no self-HTTP. Returns the bot and its poll-loop
    task (the app's lifespan cancels the task on shutdown), or `(None, None)` when
    no token is set.
    """
    token = settings.telegram_bot_token
    if not token:
        return None, None

    on_message, on_reset, on_cancel = build_telegram_callbacks(turn)
    bot = TelegramBot(
        token,
        settings.telegram_allowed_chat_ids,
        on_message,
        on_reset,
        on_cancel,
        settings_ops=settings_ops,
        approval_ops=approval_ops,
        stream=settings.telegram_stream,
    )
    return bot, asyncio.create_task(bot.run())


__all__ = ["build_approval_ops", "build_settings_ops", "start_bot"]
