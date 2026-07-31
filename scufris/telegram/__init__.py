"""Telegram frontend: a thin async httpx long-poll Bot API client.

The package splits into transport, command handling, and rendering:

| Module | Owns |
|--------|------|
| `contracts` | the injected callbacks, settings providers and approval providers |
| `text` | the Bot API limits, callback codes and every fixed operator-facing line |
| `render` | the pure renderers: turn replies, `/settings`, `/stats`, approvals |
| `api` | `BotApi` - the Bot API wire calls, the client and the poll cursor |
| `turn` | laying one streamed orchestrator turn out over messages |
| `approvals` | every host-decision path, in one place |
| `bot` | `TelegramBot` - the poll loop, the chat allowlist, command dispatch |

The chat-id allowlist IS the auth: there is no public webhook, the bot polls
outward, and an allowlisted chat is the operator. It is checked in `bot` before
an update reaches a handler, and again app-side inside the approval providers.

This module is the package's public surface; the submodules import each other
directly rather than through it.
"""

from __future__ import annotations

from .bot import TelegramBot
from .contracts import (
    ApprovalOps,
    ApprovalOutcome,
    OnCancel,
    OnMessageStream,
    OnReset,
    OrchestratorInfo,
    SettingsOps,
)
from .render import (
    _format_reasoning,
    _format_tool,
    approval_keyboard,
    markdown_reply,
    render_approval,
    render_health,
    render_reply,
    render_settings_summary,
    render_stats,
    render_tools,
    render_usage,
)
from .text import (
    APPROVALS_UNAVAILABLE,
    BUSY_REPLY,
    CANCELLED_REPLY,
    EMPTY_REPLY,
    HELP_TEXT,
    IDLE_CANCEL_REPLY,
    MAX_MESSAGE,
    MAX_TRACKED_ACTIONS,
    NO_APPROVALS,
    NOT_YOURS,
    ONE_WAY_ARMED,
    RESET_REPLY,
    SETTINGS_USAGE,
    _command_arg,
    _command_of,
)

__all__ = [
    "APPROVALS_UNAVAILABLE",
    "BUSY_REPLY",
    "CANCELLED_REPLY",
    "EMPTY_REPLY",
    "HELP_TEXT",
    "IDLE_CANCEL_REPLY",
    "MAX_MESSAGE",
    "MAX_TRACKED_ACTIONS",
    "NOT_YOURS",
    "NO_APPROVALS",
    "ONE_WAY_ARMED",
    "RESET_REPLY",
    "SETTINGS_USAGE",
    "ApprovalOps",
    "ApprovalOutcome",
    "OnCancel",
    "OnMessageStream",
    "OnReset",
    "OrchestratorInfo",
    "SettingsOps",
    "TelegramBot",
    "_command_arg",
    "_command_of",
    "_format_reasoning",
    "_format_tool",
    "approval_keyboard",
    "markdown_reply",
    "render_approval",
    "render_health",
    "render_reply",
    "render_settings_summary",
    "render_stats",
    "render_tools",
    "render_usage",
]
