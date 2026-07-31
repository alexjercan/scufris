"""The host-decision surface: every approval path the bot offers, in one module.

Announcements, the `/approvals` queue, the inline-keyboard taps, `/deny` and its
force-reply reason prompt, and the scheduled digest all live here, because a
decision that can destroy something is one surface with one set of paths - not a
rendering concern to be spread across the modules that draw messages.

There is still no second set of RULES: every decision goes through the app's one
`HostApprovalService` behind ``ApprovalOps``, and this surface supplies only WHO
is deciding, as a chat id. The chat-id allowlist that makes a chat the operator
is enforced by the bot before an update reaches here, and AGAIN app-side inside
the providers.

State is bounded on both maps: this process is long-lived (the bot restarts with
the app), and a decision surface must not accumulate state without a ceiling.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any

from ..host_actions import HostActionRecord
from .api import BotApi
from .contracts import ApprovalOps, ApprovalOutcome
from .render import approval_keyboard, confirm_keyboard, render_approval
from .text import (
    APPROVALS_UNAVAILABLE,
    CB_ABORT,
    CB_APPROVE,
    CB_CONFIRM,
    CB_DENY,
    DENY_PROMPT,
    DENY_USAGE,
    MAX_MESSAGE,
    MAX_TRACKED_ACTIONS,
    NO_APPROVALS,
    NOT_YOURS,
    ONE_WAY_ARMED,
    REASON_STILL_WANTED,
    _command_of,
)

logger = logging.getLogger(__name__)


def _toast(outcome: ApprovalOutcome) -> str:
    """The one-line answer shown on the tapped button itself.

    Short by necessity (Telegram truncates a callback answer), so the full sentence
    still goes to the chat as a message - a refusal the operator cannot read is the
    same as no refusal.
    """
    if outcome.ok:
        return "approved"
    flat = " ".join(outcome.message.split())
    return flat[:180]


class ApprovalSurface:
    """The bot's host-decision paths, over one ``BotApi`` and one allowlist.

    ``ops`` is None when this bot has no approval surface at all (nothing to
    decide through it). The app always passes one; a test may not.
    """

    def __init__(
        self, api: BotApi, ops: ApprovalOps | None, allowed: frozenset[int]
    ) -> None:
        self._api = api
        self._ops = ops
        self._allowed = allowed
        # Which messages announced which action, so a decision made ANYWHERE can
        # update what this chat is looking at: action_id -> [(chat_id, message_id)].
        # The oldest entries go first, as in `HostActionStore`.
        self._announced: "OrderedDict[str, list[tuple[int, int]]]" = OrderedDict()
        # A force-reply prompt awaiting a denial reason: (chat_id, prompt_message_id)
        # -> action_id. Keyed by the PROMPT so two pending denials in one chat cannot
        # be confused, and so a reply to something else is not read as a reason. A
        # Deny tap whose prompt is never answered would otherwise leave its entry
        # forever.
        self._reason_prompts: "OrderedDict[tuple[int, int], str]" = OrderedDict()

    def _remember(self, action_id: str, chat_id: int, message_id: int) -> None:
        """Record where an action is displayed, so a decision can update it."""
        seen = self._announced.setdefault(action_id, [])
        if (chat_id, message_id) not in seen:
            seen.append((chat_id, message_id))
        self._announced.move_to_end(action_id)
        while len(self._announced) > MAX_TRACKED_ACTIONS:
            self._announced.popitem(last=False)

    def _await_reason(self, chat_id: int, prompt_id: int, action_id: str) -> None:
        """Record that a reply to this prompt is a denial reason for this action."""
        self._reason_prompts[(chat_id, prompt_id)] = action_id
        while len(self._reason_prompts) > MAX_TRACKED_ACTIONS:
            self._reason_prompts.popitem(last=False)

    async def announce_proposal(self, record: HostActionRecord) -> None:
        """Tell every allowlisted chat that a host action is waiting for a decision.

        Called app-side when a proposal enters the queue, so the operator learns
        about it without opening anything. The message body is the shared renderer
        and the keyboard is the decision; both are re-derived on every later edit, so
        what the chat shows can never drift from what the record says.
        """
        if self._ops is None:
            return
        for chat_id in sorted(self._allowed):
            message_id = await self._api.send_message(
                chat_id,
                render_approval(record),
                reply_markup=approval_keyboard(record),
            )
            if message_id is not None:
                self._remember(record.proposal.id, chat_id, message_id)

    async def announce_decision(self, record: HostActionRecord) -> None:
        """Update what the chat is looking at after a decision - from EITHER surface.

        An action approved on the dashboard must not still be offering an Approve
        button here, and an applied result is news the chat should carry. The
        message is re-rendered from the record (so it now states the decision, and
        the result once there is one) and the keyboard is dropped, because there is
        nothing left to decide.
        """
        for chat_id, message_id in self._announced.get(record.proposal.id, []):
            await self._api.edit_message(
                chat_id,
                message_id,
                render_approval(record),
                reply_markup={"inline_keyboard": []},
            )

    async def send_digest(self, text: str) -> str:
        """Send a scheduled digest to every allowlisted chat. Returns "" or the error.

        The digest is already stored before this is called, so a failure here costs
        the MESSAGE and not the record: the caller writes the reason onto the digest
        and the schedule, and the /host/ page still shows what the checks found.
        Sent as plain text, like the approval messages - it is preformatted, and a
        parse mode would either mangle it or reject it.
        """
        errors: list[str] = []
        for chat_id in sorted(self._allowed):
            try:
                await self._api.send_message(chat_id, text[:MAX_MESSAGE])
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                logger.warning("digest delivery to chat %s failed: %s", chat_id, exc)
                errors.append(f"chat {chat_id}: {type(exc).__name__}")
        if not self._allowed:
            return "no chat is allowlisted"
        return "; ".join(errors)

    async def handle_approvals(self, chat_id: int) -> None:
        """`/approvals` - what is waiting for you right now, with its buttons."""
        if self._ops is None:
            await self._api.send_message(chat_id, APPROVALS_UNAVAILABLE)
            return
        pending = await self._ops.pending()
        if not pending:
            await self._api.send_message(chat_id, NO_APPROVALS)
            return
        for record in pending:
            message_id = await self._api.send_message(
                chat_id,
                render_approval(record),
                reply_markup=approval_keyboard(record),
            )
            if message_id is not None:
                self._remember(record.proposal.id, chat_id, message_id)

    async def handle_deny_command(self, chat_id: int, text: str) -> None:
        """`/deny <id> <reason...>` - the typed form of a denial with a reason.

        The id may be a PREFIX of the action id (they are 32 hex characters, and
        nobody is retyping that from a phone); it must match exactly one pending
        action, because denying the wrong root command because two ids shared three
        characters is not a mistake worth enabling.
        """
        if self._ops is None:
            await self._api.send_message(chat_id, APPROVALS_UNAVAILABLE)
            return
        parts = text.strip().split(maxsplit=2)
        if len(parts) < 2:
            await self._api.send_message(chat_id, DENY_USAGE)
            return
        prefix = parts[1].lower()
        reason = parts[2].strip() if len(parts) > 2 else ""
        pending = await self._ops.pending()
        matches = [r for r in pending if r.proposal.id.lower().startswith(prefix)]
        if not matches:
            await self._api.send_message(
                chat_id, f"no pending action starts with {prefix}"
            )
            return
        if len(matches) > 1:
            await self._api.send_message(
                chat_id,
                f"{len(matches)} pending actions start with {prefix}; "
                "use more characters of the id",
            )
            return
        await self._deny_with_reason(chat_id, matches[0].proposal.id, reason)

    async def _deny_with_reason(
        self, chat_id: int, action_id: str, reason: str
    ) -> None:
        """Deny through the ONE service, then report and update the message."""
        if self._ops is None:
            return
        outcome = await self._ops.deny(action_id, chat_id, reason.strip())
        await self._api.send_message(chat_id, outcome.message)
        if outcome.record is not None:
            await self.announce_decision(outcome.record)

    async def handle_reason_reply(
        self, chat_id: int, prompt_id: int, text: str
    ) -> bool:
        """Take a reply to a "why?" prompt as a denial reason. True if consumed.

        A reply that is a COMMAND is a command: an operator who answers the prompt
        with /cancel means to cancel something, not to deny with the reason
        "/cancel". The prompt stays open, so the reason can still be given, and the
        message goes on to ordinary command dispatch.
        """
        key = (chat_id, prompt_id)
        action_id = self._reason_prompts.get(key)
        if action_id is None:
            return False
        if _command_of(text):
            await self._api.send_message(chat_id, REASON_STILL_WANTED)
            return False
        self._reason_prompts.pop(key, None)
        await self._deny_with_reason(chat_id, action_id, text)
        return True

    async def handle_callback(self, callback: dict[str, Any]) -> None:
        """One inline-keyboard tap.

        Every path answers the callback query, or the client spins forever on a
        button that already did its work. The allowlist applies exactly as it does
        to a message: a tap from a chat that may not decide is answered and
        dropped, never acted on.
        """
        callback_id = callback.get("id")
        data = callback.get("data")
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        message_id = message.get("message_id")
        if not isinstance(callback_id, str) or not isinstance(data, str):
            logger.debug("telegram ignoring a malformed callback_query")
            return
        if not isinstance(chat_id, int) or chat_id not in self._allowed:
            logger.info("ignoring telegram callback from disallowed chat %s", chat_id)
            await self._api.answer_callback(callback_id, NOT_YOURS)
            return
        if self._ops is None:
            await self._api.answer_callback(callback_id, APPROVALS_UNAVAILABLE)
            return
        verb, _, action_id = data.partition(":")
        if not action_id:
            await self._api.answer_callback(callback_id, "unreadable button")
            return
        # Remember where this button lives, so a decision made on the dashboard can
        # still update it (a /approvals listing and an announcement both register,
        # but a bot restarted since the announcement has not).
        if isinstance(message_id, int):
            self._remember(action_id, chat_id, message_id)

        record = await self._ops.get(action_id)
        if record is None:
            await self._api.answer_callback(callback_id, "this action is gone")
            return

        if verb == CB_ABORT:
            # Back out of an armed one-way confirmation: nothing was decided.
            await self._api.answer_callback(callback_id, "not approved")
            if isinstance(message_id, int):
                await self._api.edit_message(
                    chat_id,
                    message_id,
                    render_approval(record),
                    reply_markup=approval_keyboard(record),
                )
            return

        if verb == CB_APPROVE and record.confirmation.style == "one_way":
            # The FIRST tap of a one-way action only arms it. Nothing is approved
            # here, and the message says what the second tap means.
            await self._api.answer_callback(callback_id, "this cannot be undone")
            if isinstance(message_id, int):
                await self._api.edit_message(
                    chat_id,
                    message_id,
                    f"{render_approval(record)}\n\n{ONE_WAY_ARMED}",
                    reply_markup=confirm_keyboard(record),
                )
            return

        if verb in (CB_APPROVE, CB_CONFIRM):
            # The acknowledgement comes from the RECORD, never from the payload: a
            # tapped button cannot assert its own terms.
            acknowledge = record.confirmation.acknowledge if verb == CB_CONFIRM else ""
            outcome = await self._ops.approve(action_id, chat_id, acknowledge)
            await self._api.answer_callback(callback_id, _toast(outcome))
            await self._api.send_message(chat_id, outcome.message)
            if outcome.record is not None:
                await self.announce_decision(outcome.record)
            return

        if verb == CB_DENY:
            # Ask for a reason rather than swallowing one: it is what reaches the
            # agent that asked, and an agent denied without a reason retries blindly.
            await self._api.answer_callback(callback_id, "why not?")
            prompt_id = await self._api.send_message(
                chat_id,
                DENY_PROMPT,
                reply_markup={"force_reply": True},
            )
            if prompt_id is not None:
                self._await_reason(chat_id, prompt_id, action_id)
            return

        await self._api.answer_callback(callback_id, "unknown button")
