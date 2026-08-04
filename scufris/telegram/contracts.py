"""The injected contracts the Telegram surface is driven through.

Plain callables and frozen dataclasses with no ``app`` import, so the transport
can be constructed and tested without the application it fronts. Everything
here is wired app-side to the SAME in-process readers and services the web
endpoints use - there is no self-HTTP and no second set of rules.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from scufris_host import HostStats
from scufris_hostctl import HostActionRecord

from ..agent import StreamEvent
from ..backends.base import Capability
from ..health import AgentHealth
from ..mcp_models import AgentTool
from ..sessions import UsageQuota

# Drive one orchestrator turn from a user message, STREAMING its events. The bot
# renders the events; the callback owns launching the turn and mapping any
# app-level condition (disabled/busy/failure) to a terminal ``StreamError`` whose
# ``detail`` is the friendly, user-facing line.
OnMessageStream = Callable[[str], AsyncIterator[StreamEvent]]
# Reset the orchestrator session (the `/new` command).
OnReset = Callable[[], Awaitable[None]]
# Cancel the orchestrator's current in-flight turn (the `/cancel` command).
OnCancel = Callable[[], Awaitable[bool]]


@dataclass(frozen=True)
class OrchestratorInfo:
    """The orchestrator's effective config, for the `/settings` summary.

    A NEUTRAL snapshot (no app import) so the transport can render it without a
    cycle back through ``app``. Built app-side from ``AgentDiagnostics.account``
    plus the two facts ``AccountInfo`` does not carry, so the `/settings` summary
    and `/settings usage` answer from ONE service call rather than two."""

    backend: str
    model: str
    auth_mode: str | None  # None for a backend with no login (mock)
    enabled: bool
    permission_mode: str
    # The account quota in its envelope, NOT unwrapped: only codex has a reader,
    # and "this backend has no quota" must not render as "the quota is empty".
    # `render` turns the three states into the three readings.
    quota: Capability[UsageQuota]


@dataclass(frozen=True)
class SettingsOps:
    """Read-only data providers behind the `/settings` and `/stats` commands.

    Each is an async callable returning a domain model the bot RENDERS (rendering
    lives in `render`). They are wired app-side to the SAME in-process readers the
    web settings endpoints use (the diagnostics service, the orchestrator tool
    catalog, the host collector) - no self-HTTP. Injected into ``TelegramBot``
    alongside the turn callbacks."""

    info: Callable[[], Awaitable[OrchestratorInfo]]
    health: Callable[[], Awaitable[AgentHealth]]
    tools: Callable[[], Awaitable[list[AgentTool]]]
    stats: Callable[[], Awaitable[HostStats]]


# --- host approvals ---------------------------------------------------------
#
# The SECOND approval surface. There is no second set of RULES: every decision
# goes through the app's one `HostApprovalService` behind these providers, and the
# only thing this surface supplies is WHO is deciding - which it does by handing
# over the chat id, never an actor string it made up.
#
# The credential is the allowlist: an allowlisted chat IS the operator. That is
# checked in the bot's update handling and AGAIN app-side inside these providers,
# so neither layer is the only thing standing between a stray chat and a root
# command.


@dataclass(frozen=True)
class ApprovalOutcome:
    """What came of a decision: whether it happened, and what to tell the operator.

    The MESSAGE comes from the app - which means from the service's own refusals
    ("already denied by ...", "this proposal has expired", "needs the explicit
    acknowledgement ...") rather than from anything this transport invents. That is
    what keeps the two surfaces saying the same thing about the same rule.
    """

    ok: bool
    message: str
    record: HostActionRecord | None = None


@dataclass(frozen=True)
class ApprovalOps:
    """The host-approval providers behind the bot's queue, buttons and commands.

    Wired app-side to the SAME `HostApprovalService` the web routes call - no
    self-HTTP, and no rule of its own. Each decision takes the CHAT ID; the app
    turns that into the audited actor (``operator:telegram:<chat_id>``) and refuses
    a chat that is not allowlisted.
    """

    # The proposals still waiting for a decision, newest first.
    pending: Callable[[], Awaitable[list[HostActionRecord]]]
    # One action by id, or None if this server has never heard of it.
    get: Callable[[str], Awaitable[HostActionRecord | None]]
    # (action_id, chat_id, acknowledge) - the acknowledgement is empty for an
    # ordinary approval and carries the token for a one-way one.
    approve: Callable[[str, int, str], Awaitable[ApprovalOutcome]]
    # (action_id, chat_id, reason) - the reason reaches the agent that asked.
    deny: Callable[[str, int, str], Awaitable[ApprovalOutcome]]
