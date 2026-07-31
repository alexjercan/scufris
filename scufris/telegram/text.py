"""The literal surface of the bot: Bot API limits, callback codes, and every
fixed line the operator reads, plus the small text helpers that parse a command
out of a message.

Kept apart from `render` because these strings are read by the command handling
and the approval surface too, not only by the renderers.
"""

from __future__ import annotations

DEFAULT_API_BASE = "https://api.telegram.org"

# How many actions the bot tracks message ids / open reason prompts for. Bounded so
# a long-lived process cannot accumulate them; well above any real queue (the app's
# own registry caps at 200).
MAX_TRACKED_ACTIONS = 200

# Telegram's hard per-message cap. A host action's rendered text can exceed it (a
# closure diff is long), and a 400 from the Bot API would mean the operator never
# sees the proposal at all - so it is trimmed. WHERE it is trimmed is the whole
# point: see `render.render_approval`.
MAX_MESSAGE = 4096

# `callback_data` is capped at 64 BYTES by the Bot API, so the payload is a short
# verb plus the action id (32 hex chars) and nothing else. In particular the
# acknowledgement token is NOT carried here: the bot re-reads the record and takes
# the token from it, so a tapped button can never assert its own terms.
CB_APPROVE = "ha"
CB_CONFIRM = "hk"
CB_DENY = "hd"
CB_ABORT = "hx"

# What the operator is told when a decision cannot be made, or has already been
# made. Every OTHER message about a decision comes from the app (which is to say
# from the approval service's own refusals), so these are only the cases the
# transport itself answers.
APPROVALS_UNAVAILABLE = (
    "host approvals are not available on this server (no privileged helper configured)"
)
NO_APPROVALS = "nothing is waiting for your decision"
REASON_STILL_WANTED = (
    "that looks like a command, so it was not taken as the reason - reply again with "
    "why you are refusing, or send - for no reason"
)
NOT_YOURS = "this chat cannot decide host actions"
DENY_USAGE = "usage: /deny <action-id> <reason>  (the reason reaches the agent)"
DENY_PROMPT = (
    "Why not? Reply to this message with the reason - it reaches the agent that "
    "asked, so it can adapt instead of asking again. Reply with - for no reason."
)
ONE_WAY_ARMED = (
    "THIS CANNOT BE UNDONE. Tap the confirm button to approve it anyway, or Back "
    "to leave it pending."
)

HELP_TEXT = (
    "Scufris orchestrator bot. Commands:\n"
    "/new (or /reset) - start a fresh conversation (forget context)\n"
    "/cancel - stop the current message\n"
    "/approvals - host actions waiting for your decision\n"
    "/deny <id> <reason> - refuse a pending host action, with a reason\n"
    "/settings - orchestrator config summary\n"
    "/settings health - backend + MCP diagnostics\n"
    "/settings usage - account usage/quota\n"
    "/settings tools - the orchestrator tool catalog\n"
    "/stats - a compact host health snapshot\n"
    "/help - show this message\n"
    "\n"
    "Any other message is forwarded to the orchestrator."
)

RESET_REPLY = "Started a fresh conversation."
CANCELLED_REPLY = "Cancelled current message."
IDLE_CANCEL_REPLY = "No active message to cancel."
BUSY_REPLY = "I'm still working on the previous message - try again in a moment."
SETTINGS_USAGE = "Usage: /settings [health|usage|tools]"

# Shown when a turn produces no final text (Telegram rejects an empty message).
EMPTY_REPLY = "(the orchestrator returned no text)"


def _preview(text: str, limit: int = 80) -> str:
    """A short, single-line preview of a message for DEBUG logs (never the full
    body, which can be long and may hold sensitive content)."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _command_of(text: str) -> str:
    """The leading bot command of a message, lower-cased and de-mentioned.

    Telegram sends group commands as ``/new@mybot``; strip the ``@mention`` so
    the command matches regardless of how the client addressed the bot. A
    message that does not start with a command returns "".
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return ""
    token = stripped.split(maxsplit=1)[0]
    return token.split("@", 1)[0].lower()


def _command_arg(text: str) -> str:
    """The first argument after the command, lower-cased (the `/settings` sub).

    ``/settings health`` -> "health"; a bare ``/settings`` (or extra words after
    the first) yields "" / just the first token. Empty when there is no argument.
    """
    parts = text.strip().split()
    return parts[1].lower() if len(parts) > 1 else ""
