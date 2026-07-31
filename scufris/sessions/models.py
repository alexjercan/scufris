"""The session data models, shared by the readers and by the backends.

This module imports nothing from ``scufris`` beyond ``config``, so ``agent`` and
``backends`` can depend on it without a cycle.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RateWindow(BaseModel):
    """One rate-limit window (codex reports a weekly primary at 10080 minutes)."""

    used_percent: float
    window_minutes: int
    resets_at: int | None = None


class UsageQuota(BaseModel):
    """Account-wide subscription usage, as codex last reported it."""

    plan_type: str | None = None
    primary: RateWindow | None = None
    secondary: RateWindow | None = None


class ToolCall(BaseModel):
    """One tool the agent invoked in a turn (server + tool name + status)."""

    server: str
    tool: str
    status: str


class TokenUsage(BaseModel):
    """Token counts for a request/turn, as codex reports them."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0


class SessionInfo(BaseModel):
    """One codex session, for the switch list."""

    id: str
    title: str
    started_at: datetime | None = None
    updated_at: datetime | None = None
    git_branch: str | None = None
    cwd: str | None = None


class SessionContext(BaseModel):
    """A snapshot of one session's context usage.

    Not a per-component ``/context`` breakdown - codex does not expose that. These
    are the real axes it does give. ``input_tokens``/``cached_input_tokens``
    describe the CURRENT context occupancy (the last request's input), so
    ``input_tokens / context_window`` is a truthful "how full" figure;
    ``output``/``reasoning``/``total`` are cumulative work over the session.
    """

    session_id: str
    context_window: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0
    turn_count: int = 0
    tool_call_count: int = 0


class TranscriptMessage(BaseModel):
    """One message in a session transcript, for re-rendering on switch."""

    role: str  # "user" | "assistant"
    text: str
    ts: datetime | None = None  # when the turn was recorded, for a UI timestamp
    # An assistant message carries the tools it ran and the turn's token usage, so
    # the UI can rebuild the "ran <tool>" chips + token count on reload (they would
    # otherwise render only on the live turn). Empty/None for user messages.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage | None = None
    # Codex "thinking" (reasoning) that streamed live during the turn, recovered
    # from the reasoning sidecar (reasoning is NOT in the rollout - only an
    # encrypted blob). None for user turns, non-codex turns, and turns the sidecar
    # does not cover. Merged in by CodexBackend.read_transcript, not read here.
    reasoning: str | None = None
