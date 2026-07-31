"""Re-rendering a session's conversation: read it, merge reasoning, seed a fork.

``read_transcript`` reconstructs the turns from the rollout; ``merge_reasoning``
attaches the reasoning sidecar's text to the turns it genuinely covers; and
``format_fork_seed`` turns prior turns back into a prompt. The last two are pure,
so their alignment and capping rules are unit-testable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .models import TokenUsage, ToolCall, TranscriptMessage
from .rollout import (
    _event_kind,
    _find_rollout,
    _int,
    _iter_events,
    _parse_ts,
    _payload,
)
from .steering import strip_steering


def _tool_call_from_end(payload: dict[str, Any]) -> ToolCall | None:
    """A ``ToolCall`` from an ``mcp_tool_call_end`` payload, or None if malformed.

    codex records the result as a Rust-style enum (``{"Ok": ...}`` / ``{"Err":
    ...}``), so success is "Ok present".
    """
    inv = payload.get("invocation")
    if not isinstance(inv, dict):
        return None
    server = inv.get("server")
    tool = inv.get("tool")
    if not isinstance(server, str) or not isinstance(tool, str):
        return None
    result = payload.get("result")
    ok = isinstance(result, dict) and "Ok" in result
    return ToolCall(server=server, tool=tool, status="completed" if ok else "error")


def _last_usage(payload: dict[str, Any]) -> TokenUsage | None:
    """The per-request usage from a ``token_count`` payload's ``last_token_usage``."""
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    last = info.get("last_token_usage")
    if not isinstance(last, dict):
        return None
    return TokenUsage(
        input_tokens=_int(last.get("input_tokens")),
        cached_input_tokens=_int(last.get("cached_input_tokens")),
        output_tokens=_int(last.get("output_tokens")),
        reasoning_output_tokens=_int(last.get("reasoning_output_tokens")),
    )


def read_transcript(
    codex_home: Path, session_id: str | None, limit: int = 200
) -> list[TranscriptMessage]:
    """The session's conversation, so switching to it can re-render its history.

    User turns come from ``user_message``; assistant turns from the
    ``agent_message`` final answer (intermediate reasoning phases are skipped). A
    turn's ``mcp_tool_call_end`` events (which sit between the commentary and the
    final answer) are attached to that final answer as ``tool_calls``, and the
    turn's output tokens (the ``token_count`` right after the final answer) as
    ``usage`` - so the UI rebuilds the "ran <tool>" chips + token count on reload.
    Capped at ``limit`` messages (most recent kept).
    """
    if not session_id:
        return []
    path = _find_rollout(codex_home, session_id)
    if path is None:
        return []
    messages: list[TranscriptMessage] = []
    pending_tools: list[ToolCall] = []
    awaiting_usage: TranscriptMessage | None = None
    for event in _iter_events(path):
        kind = _event_kind(event)
        ts = _parse_ts(event.get("timestamp"))
        payload = _payload(event)
        if kind == "user_message":
            # A new turn - tool calls belong to the turn they ran in, not the next.
            pending_tools = []
            awaiting_usage = None
            text = payload.get("message")
            if isinstance(text, str):
                # Hide the injected steering preamble in the re-rendered history.
                text = strip_steering(text).strip()
            if isinstance(text, str) and text:
                messages.append(TranscriptMessage(role="user", text=text, ts=ts))
        elif kind == "mcp_tool_call_end":
            call = _tool_call_from_end(payload)
            if call is not None:
                pending_tools.append(call)
        elif kind == "agent_message":
            if payload.get("phase") not in (None, "final_answer"):
                continue
            text = payload.get("message")
            if isinstance(text, str) and text.strip():
                message = TranscriptMessage(
                    role="assistant",
                    text=text.strip(),
                    ts=ts,
                    tool_calls=pending_tools,
                )
                messages.append(message)
                pending_tools = []
                # Its output tokens arrive in the NEXT token_count event.
                awaiting_usage = message
        elif kind == "token_count":
            if awaiting_usage is not None and awaiting_usage.usage is None:
                awaiting_usage.usage = _last_usage(payload)
                awaiting_usage = None
    return messages[-limit:]


def reasoning_fingerprint(text: str) -> str:
    """A short, whitespace-normalized fingerprint of an assistant answer.

    Used to guard the reasoning sidecar's alignment to the transcript (see
    ``merge_reasoning``): the streamed final answer and codex's on-disk recorded
    answer can differ in trailing/collapsed whitespace, so normalize before
    hashing. This is an alignment key, not a security digest - a short prefix of
    the hash is enough.
    """
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def merge_reasoning(messages: list[TranscriptMessage], entries: list[Any]) -> None:
    """Attach sidecar reasoning to the assistant messages it covers, IN PLACE.

    ``entries`` are the sidecar's per-turn records (``answer`` fingerprint +
    ``reasoning`` text), oldest->newest - one per completed assistant turn, the
    same order assistant messages appear in ``messages``. Pair them from the tail
    (newest) backwards; accept a pair only while the answer fingerprint matches,
    and stop at the first mismatch. That way a PARTIAL or pre-existing sidecar
    (feature deployed mid-session, or a turn scufris never captured) attaches
    reasoning only to the turns it genuinely covers, and a gross mismatch yields
    no reasoning rather than a mislabeled spoiler. Pure (no I/O), so the tail
    alignment is unit-testable.
    """
    if not entries:
        return
    assistant = [m for m in messages if m.role == "assistant"]
    # strict=False on purpose: a partial/pre-existing sidecar has FEWER entries
    # than assistant messages, and tail-alignment must stop at the shorter one.
    for msg, entry in zip(reversed(assistant), reversed(entries), strict=False):
        if reasoning_fingerprint(msg.text) != entry.answer:
            break
        if entry.reasoning:
            msg.reasoning = entry.reasoning


# How many prior turns to paste as context when forking. codex-exec has no native
# "branch at turn N", so a fork re-seeds a fresh session with the earlier turns as
# text; cap it so a long history does not blow up the seed prompt.
FORK_CONTEXT_TURNS = 40


def format_fork_seed(
    context: list[TranscriptMessage],
    text: str,
    max_turns: int = FORK_CONTEXT_TURNS,
) -> str:
    """Build the seed prompt for a fork: prior turns as context + the edited text.

    Pure (no I/O) so it is unit-testable. When there is no prior context, the seed
    is just the edited text - forking the very first message is a plain new chat.
    """
    text = text.strip()
    kept = context[-max_turns:] if max_turns > 0 else context
    if not kept:
        return text
    lines = [
        "The following is an earlier conversation, provided for context.",
        "",
    ]
    for message in kept:
        who = "User" if message.role == "user" else "Assistant"
        lines.append(f"{who}: {message.text}")
    lines += [
        "",
        "End of earlier context. Continue the conversation from my next message.",
        "",
        text,
    ]
    return "\n".join(lines)
