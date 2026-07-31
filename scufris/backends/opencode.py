"""The opencode adapter: drives a running `opencode serve` daemon over HTTP.

opencode's headless surface is an HTTP daemon, so this backend uses an async HTTP
client rather than a stdio subprocess - structurally the codex ``app_server``
shape, not the claude stdin one. A turn runs SYNCHRONOUSLY (``send_message``
blocks and returns the whole ``{info, parts}`` reply); read_status and
read_transcript read the session back over ``GET /session/{id}/message``.

TODO: stream tokens live over the daemon's `/event` SSE bus.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator

import httpx

from ..agent import (
    AgentReply,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamTextDelta,
    StreamTool,
    ToolCall,
)
from ..config import Settings
from ..logsetup import truncate
from ..opencode_client import (
    Message,
    ModelRef,
    OpencodeClient,
    OpencodeError,
    OpencodeStaleSessionError,
    SendMessageRequest,
    TextPartInput,
)
from ..sessions import SessionContext, TokenUsage, TranscriptMessage
from .base import _LAST_MESSAGE_PREVIEW, BackendStatus, _context_from_status

# Permission mode -> opencode per-request `tools` enable/disable map. opencode's
# approval flow ("ask") has no answerer on a headless server, so the safe lever is
# tool AVAILABILITY: a disabled tool cannot be called. manual disables all mutating
# tools (read-only); edit allows edits but not shell; auto leaves everything on
# (empty map).
_OPENCODE_MUTATING_TOOLS = ("edit", "write", "patch", "bash")
_OPENCODE_PERMISSION: dict[str, dict[str, bool]] = {
    "manual": {tool: False for tool in _OPENCODE_MUTATING_TOOLS},
    "edit": {"bash": False},
    "auto": {},
}


def _opencode_tools_for(mode: str) -> dict[str, bool]:
    return _OPENCODE_PERMISSION.get(mode, _OPENCODE_PERMISSION["manual"])


def _opencode_ms_to_dt(ms: int | None) -> "datetime | None":
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, timezone.utc)


def _opencode_usage(msg: Message) -> TokenUsage | None:
    tokens = msg.info.tokens
    if tokens is None:
        return None
    cache = (tokens.model_extra or {}).get("cache") or {}
    cached = int(cache.get("read", 0)) if isinstance(cache, dict) else 0
    return TokenUsage(
        input_tokens=tokens.input,
        cached_input_tokens=cached,
        output_tokens=tokens.output,
        reasoning_output_tokens=tokens.reasoning,
    )


def _opencode_tool_calls(msg: Message) -> list[ToolCall]:
    return [
        ToolCall(server="opencode", tool=p.tool_name(), status="completed")
        for p in msg.tool_parts()
    ]


def parse_opencode_transcript(messages: list[Message]) -> list[TranscriptMessage]:
    """Fold opencode `{info, parts}` messages into TranscriptMessages, oldest-first.

    A user or assistant message with any text becomes a message; a tool-only
    assistant turn is kept (empty text, carrying its tool calls), mirroring the
    claude transcript parser.
    """
    out: list[TranscriptMessage] = []
    for msg in messages:
        role = msg.info.role
        if role not in ("user", "assistant"):
            continue
        text = msg.text()
        tools = _opencode_tool_calls(msg) if role == "assistant" else []
        if not text and not tools and role == "assistant":
            continue
        out.append(
            TranscriptMessage(
                role=role,
                text=text,
                ts=_opencode_ms_to_dt(msg.info.time.created if msg.info.time else None),
                tool_calls=tools,
                usage=_opencode_usage(msg) if role == "assistant" else None,
            )
        )
    return out


class OpenCodeBackend:
    """The "opencode" backend: drives a running `opencode serve` daemon over HTTP.

    ``name`` is "opencode". The daemon URL/password/model/provider come from
    ``Settings`` (``opencode_url`` etc). ``_make_client`` is a seam the tests
    monkeypatch to inject a fake client.
    """

    name: str = "opencode"

    def _make_client(self, settings: Settings) -> OpencodeClient:
        return OpencodeClient(
            settings.opencode_url,
            settings.opencode_password,
            timeout=settings.agent_timeout_seconds,
        )

    async def stream(
        self,
        settings: Settings,
        prompt: str,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        image_paths: list[str] | None = None,
        permission_mode: str = "manual",
        is_orchestrator: bool = False,
        agent_id: str = "",
    ) -> AsyncIterator[StreamEvent]:
        # cwd is not used: the daemon's working dir is fixed at `opencode serve`
        # launch, not per turn (unlike codex/claude which take cwd per subprocess).
        # TODO: support image attachments here via FilePartInput.
        request = SendMessageRequest(
            parts=[TextPartInput(text=prompt)],
            model=ModelRef(
                providerID=settings.opencode_provider, modelID=settings.opencode_model
            ),
            tools=_opencode_tools_for(permission_mode) or None,
        )
        client = self._make_client(settings)
        try:
            reply = await self._send(client, session_id, request, agent_id=agent_id)
        except OpencodeError as exc:
            yield StreamError(detail=f"opencode: {exc}")
            return
        finally:
            await client.close()

        msg, new_session_id = reply
        for call in _opencode_tool_calls(msg):
            yield StreamTool(tool=call)
        text = msg.text()
        if text:
            yield StreamTextDelta(delta=text)
        yield StreamDone(
            reply=AgentReply(
                text=text.strip(),
                status="completed",
                tool_calls=_opencode_tool_calls(msg),
                usage=_opencode_usage(msg),
            ),
            session_id=new_session_id,
        )

    async def _send(
        self,
        client: OpencodeClient,
        session_id: str | None,
        request: SendMessageRequest,
        *,
        agent_id: str = "",
    ) -> tuple[Message, str]:
        """Resolve/create a session and run one turn; recreate once on a stale id.

        A newly created session is tagged with ``metadata={"agent_id": ...}`` so
        ownership is recorded on the provider side. A resumed session is left
        untouched (it was tagged when first created)."""
        metadata = {"agent_id": agent_id} if agent_id else None
        sid = session_id or (await client.create_session(metadata=metadata)).id
        try:
            return await client.send_message(sid, request), sid
        except OpencodeStaleSessionError:
            # The id we were handed is gone (deleted, or a cross-backend id after a
            # backend switch); start a fresh session and retry once.
            fresh = (await client.create_session(metadata=metadata)).id
            return await client.send_message(fresh, request), fresh

    def read_status(
        self, settings: Settings, session_id: str | None
    ) -> BackendStatus | None:
        if not session_id:
            return None
        messages = self._read_messages(settings, session_id)
        if messages is None:
            return None
        turns = sum(1 for m in messages if m.info.role == "user")
        tool_calls = sum(len(m.tool_parts()) for m in messages)
        last_message: str | None = None
        input_tokens = output_tokens = 0
        updated_at: datetime | None = None
        for msg in messages:
            if msg.info.time and msg.info.time.created:
                updated_at = _opencode_ms_to_dt(msg.info.time.created)
            if msg.info.role == "assistant":
                if msg.text().strip():
                    last_message = truncate(msg.text().strip(), _LAST_MESSAGE_PREVIEW)
                usage = _opencode_usage(msg)
                if usage is not None:
                    input_tokens = usage.input_tokens or input_tokens
                    output_tokens = usage.output_tokens or output_tokens
        return BackendStatus(
            session_id=session_id,
            turns=turns,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_window=0,
            last_message=last_message,
            updated_at=updated_at.timestamp() if updated_at else None,
        )

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        if not session_id:
            return []
        messages = self._read_messages(settings, session_id)
        return parse_opencode_transcript(messages) if messages else []

    def read_context(
        self, settings: Settings, session_id: str | None
    ) -> SessionContext | None:
        # opencode exposes no per-session context window; map read_status.
        return _context_from_status(self.read_status(settings, session_id))

    async def delete_session(self, settings: Settings, session_id: str | None) -> bool:
        """Delete the session on the daemon via ``OpencodeClient``. Any failure ->
        False (never raises), so the registry forget is the fallback."""
        if not session_id:
            return False
        client = self._make_client(settings)
        try:
            return await client.delete_session(session_id)
        except OpencodeError:
            return False
        finally:
            await client.close()

    def _read_messages(
        self, settings: Settings, session_id: str
    ) -> list[Message] | None:
        """Fetch a session's messages via the daemon; None if it cannot be read.

        read_status/read_transcript are SYNCHRONOUS (the AgentBackend protocol),
        and their FastAPI/FastMCP handlers are sync `def` (run in a threadpool, not
        on the event loop), so a plain blocking httpx read is the simplest correct
        choice - it mirrors codex/claude's blocking file reads at the same call
        sites (app.py agent_run_status/agent_transcript, mcp_server agent_status).
        Any failure -> None (never crash a snapshot).
        """
        auth = (
            httpx.BasicAuth("", settings.opencode_password)
            if settings.opencode_password
            else None
        )
        url = f"{settings.opencode_url.rstrip('/')}/session/{session_id}/message"
        try:
            resp = httpx.get(url, auth=auth, timeout=10.0)
            if resp.status_code != 200:
                return None
            return [Message.model_validate(item) for item in resp.json()]
        except (httpx.HTTPError, ValueError):
            return None
