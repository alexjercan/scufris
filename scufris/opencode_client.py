"""Async HTTP client for a local ``opencode serve`` daemon.

The ``opencode`` agent backend (``backends.OpenCodeBackend``) drives a running
``opencode serve`` daemon over this client instead of shelling out per turn the
way codex/claude do - opencode's headless surface is an HTTP server + event bus
(the codex ``app_server`` shape), not a stdio subprocess. The daemon is itself
pointed at a self-hosted llama.cpp server via a custom OpenAI-compatible
provider (see ``examples/opencode/opencode.json`` and
``tasks/20260722-135520/NOTES.md``).

Adapted from the proven ``scufris-bot`` reference
(``scufris_server/opencode_client.py`` @ ``feature/opencode-v2``), trimmed to the
endpoints this backend uses: ``health``, ``create_session``, ``send_message``
(synchronous turn) and ``get_messages`` (for read_status / read_transcript).

Auth: ``OPENCODE_SERVER_PASSWORD`` becomes the HTTP Basic *password*; the
username is empty per opencode's serve contract. With no password (loopback
dev) no Authorization header is sent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OpencodeError(Exception):
    """Base for every error this module raises."""


class OpencodeNetworkError(OpencodeError):
    """Transport-level failure (connect/read timeout, DNS, refused, ...)."""

    def __init__(self, message: str, *, original: BaseException | None = None) -> None:
        super().__init__(message)
        self.original = original


class OpencodeStatusError(OpencodeError):
    """Base for HTTP non-2xx responses; carries status + parsed body."""

    def __init__(
        self, status_code: int, body: Any, *, message: str | None = None
    ) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(message or f"opencode returned HTTP {status_code}: {body!r}")


class OpencodeClientError(OpencodeStatusError):
    """opencode returned a 4xx - our request was malformed."""


class OpencodeServerError(OpencodeStatusError):
    """opencode returned a 5xx - opencode is broken."""


class OpencodeUnavailable(OpencodeError):
    """Cannot get a healthy response from opencode at all (network or non-200).

    Raised exclusively by :meth:`OpencodeClient.health` so liveness checks only
    need a single ``except``.
    """


class OpencodeStaleSessionError(OpencodeError):
    """A session id we thought was live returned 404 - recreate and retry once."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"opencode session {session_id!r} returned 404 (stale)")
        self.session_id = session_id


# ---------------------------------------------------------------------------
# Response models (extra="allow" so opencode can grow fields without breaking us)
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """``GET /global/health`` payload."""

    healthy: bool
    version: str


class TokenUsage(BaseModel):
    """The ``tokens`` sub-object opencode reports on an assistant message."""

    model_config = ConfigDict(extra="allow")
    input: int = 0
    output: int = 0
    reasoning: int = 0
    total: int | None = None


class MessageTime(BaseModel):
    """The ``time`` sub-object: unix-millisecond lifecycle timestamps."""

    model_config = ConfigDict(extra="allow")
    created: int | None = None
    completed: int | None = None


class Session(BaseModel):
    """``POST /session`` response (and an element of ``GET /session``)."""

    model_config = ConfigDict(extra="allow")
    id: str
    title: str | None = None


class Part(BaseModel):
    """One element of a message's ``parts`` array.

    opencode emits many ``type`` values (``text``, ``reasoning``, ``step-start``,
    ``step-finish``, ``tool``, ``tool-call``, ``tool-result``); ``text`` is
    hoisted because callers read it most. Tool parts are recognised by
    ``"tool" in type``.
    """

    model_config = ConfigDict(extra="allow")
    type: str
    text: str | None = None

    def is_tool(self) -> bool:
        return "tool" in self.type

    def tool_name(self) -> str:
        """Best-effort tool name from a tool part (falls back to the type)."""
        extra = self.model_extra or {}
        return str(extra.get("tool") or extra.get("name") or self.type)


class MessageInfo(BaseModel):
    """The ``info`` half of a message: role + per-turn metadata."""

    model_config = ConfigDict(extra="allow")
    id: str | None = None
    role: str = ""
    tokens: TokenUsage | None = None
    time: MessageTime | None = None


class Message(BaseModel):
    """One ``{info, parts}`` message (send-message response and list element)."""

    info: MessageInfo
    parts: list[Part]

    def text(self) -> str:
        """Concatenate every ``text`` part (skips reasoning/steps/tool parts)."""
        return "".join(p.text or "" for p in self.parts if p.type == "text")

    def tool_parts(self) -> list[Part]:
        return [p for p in self.parts if p.is_tool()]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ModelRef(BaseModel):
    """Identifies a model for ``send_message`` (provider id + model id)."""

    providerID: str
    modelID: str


class TextPartInput(BaseModel):
    """A single text part of the outbound user message."""

    type: Literal["text"] = "text"
    text: str


class SendMessageRequest(BaseModel):
    """Body for ``POST /session/{id}/message``. ``None`` fields are omitted.

    ``tools`` is a per-request enable/disable map (``{"edit": false, ...}``) -
    the mechanism the backend uses to enforce the permission mode headlessly (a
    disabled tool is simply unavailable; opencode's ``ask`` approval flow has no
    answerer on the server, so allow/deny-by-availability is the safe lever).
    """

    parts: list[TextPartInput]
    model: ModelRef | None = None
    agent: str | None = None
    system: str | None = None
    tools: dict[str, bool] | None = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpencodeClient:
    """Async HTTP client for the local opencode daemon."""

    def __init__(
        self,
        base_url: str,
        password: str | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        auth = httpx.BasicAuth("", password) if password else None
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, auth=auth
        )

    async def close(self) -> None:
        """Close the underlying transport. Safe to call repeatedly."""
        await self._client.aclose()

    async def __aenter__(self) -> "OpencodeClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    # ----- internals --------------------------------------------------------

    async def _request(
        self, method: str, url: str, *, json: Mapping[str, Any] | None = None
    ) -> httpx.Response:
        try:
            resp = await self._client.request(method, url, json=json)
        except httpx.RequestError as exc:
            raise OpencodeNetworkError(
                f"{method} {url}: {exc!r}", original=exc
            ) from exc
        if resp.status_code == 404:
            # Let callers distinguish a stale session from a generic 4xx.
            raise OpencodeClientError(404, self._parse_body(resp))
        if 400 <= resp.status_code < 500:
            raise OpencodeClientError(resp.status_code, self._parse_body(resp))
        if 500 <= resp.status_code < 600:
            raise OpencodeServerError(resp.status_code, self._parse_body(resp))
        return resp

    @staticmethod
    def _parse_body(resp: httpx.Response) -> Any:
        try:
            return resp.json()
        except ValueError:
            return resp.text

    # ----- typed endpoints --------------------------------------------------

    async def health(self) -> HealthResponse:
        """Probe ``GET /global/health``; any failure -> :class:`OpencodeUnavailable`."""
        try:
            resp = await self._client.get("/global/health")
        except httpx.RequestError as exc:
            raise OpencodeUnavailable(
                f"cannot reach opencode at {self._client.base_url}: {exc!r}"
            ) from exc
        if resp.status_code != 200:
            raise OpencodeUnavailable(
                f"opencode /global/health returned HTTP {resp.status_code}"
            )
        return HealthResponse.model_validate(resp.json())

    async def create_session(self, *, title: str | None = None) -> Session:
        """Create a new opencode session."""
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        resp = await self._request("POST", "/session", json=body)
        return Session.model_validate(resp.json())

    async def send_message(
        self, session_id: str, request: SendMessageRequest
    ) -> Message:
        """Send a message and block until the reply arrives (synchronous turn).

        A 404 for ``session_id`` is surfaced as :class:`OpencodeStaleSessionError`
        so the caller can recreate the session and retry.
        """
        payload = request.model_dump(exclude_none=True)
        try:
            resp = await self._request(
                "POST", f"/session/{session_id}/message", json=payload
            )
        except OpencodeClientError as exc:
            if exc.status_code == 404:
                raise OpencodeStaleSessionError(session_id) from exc
            raise
        return Message.model_validate(resp.json())

    async def get_messages(self, session_id: str) -> list[Message]:
        """List a session's messages oldest-first (``GET /session/{id}/message``)."""
        resp = await self._request("GET", f"/session/{session_id}/message")
        raw = resp.json()
        return [Message.model_validate(item) for item in raw]
