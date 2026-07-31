"""What a turn emits: the reply, the stream event union, and the read ceiling.

Every backend speaks this vocabulary, so the SSE surface is the same whichever
one ran the turn. ``ToolCall`` and ``TokenUsage`` live in ``sessions.models`` (so
a ``TranscriptMessage`` can carry them without an import cycle) and are
re-exported here for callers that read them off a live turn.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..sessions import TokenUsage, ToolCall


class AgentUnavailable(RuntimeError):
    """Raised when the agent cannot serve a request (disabled or unconfigured)."""


class AgentReply(BaseModel):
    text: str
    status: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage | None = None


# Events streamed during one turn (SSE), so the UI can show live progress. A
# `tool` fires as each MCP tool completes; `done` carries the final reply; `error`
# reports a failed turn. The `kind` field discriminates them on the wire.
class StreamTool(BaseModel):
    kind: Literal["tool"] = "tool"
    tool: ToolCall


class StreamDone(BaseModel):
    kind: Literal["done"] = "done"
    reply: AgentReply
    session_id: str | None = None


class StreamError(BaseModel):
    kind: Literal["error"] = "error"
    detail: str


# Emitted the moment a turn's session (codex thread) id is known - right after
# thread/start|resume, before the turn streams - so a client reattaching mid-turn,
# and the run-launch path, learn the session id without waiting for `done`. codex
# only; other backends carry their id on `done`. See `_stream_app_server`.
class StreamSessionStarted(BaseModel):
    kind: Literal["session_started"] = "session_started"
    session_id: str


# app-server-only: token-by-token assistant text, and reasoning ("thinking").
class StreamTextDelta(BaseModel):
    kind: Literal["text_delta"] = "text_delta"
    delta: str


class StreamReasoningDelta(BaseModel):
    kind: Literal["reasoning_delta"] = "reasoning_delta"
    delta: str


StreamEvent = (
    StreamTool
    | StreamDone
    | StreamError
    | StreamTextDelta
    | StreamReasoningDelta
    | StreamSessionStarted
)


# Max size (bytes) of a single line the JSON-RPC / stream-json readers accept from
# a backend subprocess. asyncio's StreamReader defaults to 64 KiB and raises a bare
# `ValueError` ("Separator is not found, and chunk exceed the limit") on any longer
# line - which for a codex/claude app-server frame is a real, benign occurrence: a
# single command-output notification (a big `rg`, a `tatr ls` over hundreds of
# tasks, a large file dump) easily exceeds 64 KiB. We raise the ceiling to 8 MiB so
# such frames stream through instead of erroring the run. Shared by both the codex
# app-server launch (`_stream_app_server`) and `ClaudeBackend.stream`.
STREAM_READ_LIMIT = 8 * 1024 * 1024

__all__ = [
    "STREAM_READ_LIMIT",
    "AgentReply",
    "AgentUnavailable",
    "StreamDone",
    "StreamError",
    "StreamEvent",
    "StreamReasoningDelta",
    "StreamSessionStarted",
    "StreamTextDelta",
    "StreamTool",
    "TokenUsage",
    "ToolCall",
]
