"""Tests for scufris.opencode_client - the async HTTP client for opencode serve.

Uses respx to stub the httpx transport; no opencode process needed. Payload
shapes mirror the live daemon probed in tasks/20260722-135520 (a message is
``{info, parts}`` with info.role + info.tokens + info.time, parts a list of
typed fragments).
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
import respx

from scufris.opencode_client import (
    Message,
    ModelRef,
    OpencodeClient,
    OpencodeClientError,
    OpencodeNetworkError,
    OpencodeServerError,
    OpencodeStaleSessionError,
    OpencodeUnavailable,
    SendMessageRequest,
    TextPartInput,
)

BASE = "http://opencode.test"


def _assistant_message(text: str = "hello from gemma") -> dict[str, Any]:
    return {
        "info": {
            "id": "msg_a",
            "role": "assistant",
            "tokens": {
                "input": 100,
                "output": 6,
                "reasoning": 0,
                "total": 106,
                "cache": {"read": 40, "write": 0},
            },
            "time": {"created": 1784722453454, "completed": 1784722534514},
        },
        "parts": [
            {"type": "step-start"},
            {"type": "text", "text": text},
            {"type": "step-finish"},
        ],
    }


async def test_health_ok() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.get("/global/health").mock(
            return_value=httpx.Response(
                200, json={"healthy": True, "version": "1.17.9"}
            )
        )
        async with OpencodeClient(BASE) as client:
            health = await client.health()
    assert health.healthy is True
    assert health.version == "1.17.9"


async def test_health_non_200_is_unavailable() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.get("/global/health").mock(return_value=httpx.Response(503))
        async with OpencodeClient(BASE) as client:
            with pytest.raises(OpencodeUnavailable):
                await client.health()


async def test_health_network_error_is_unavailable() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.get("/global/health").mock(side_effect=httpx.ConnectError("refused"))
        async with OpencodeClient(BASE) as client:
            with pytest.raises(OpencodeUnavailable):
                await client.health()


async def test_auth_header_sent_when_password_set() -> None:
    seen: dict[str, Any] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"healthy": True, "version": "1.17.9"})

    with respx.mock(base_url=BASE) as mock:
        mock.get("/global/health").mock(side_effect=_capture)
        async with OpencodeClient(BASE, password="hunter2") as client:
            await client.health()
    # HTTP Basic with an EMPTY username and the password.
    assert seen["auth"] == "Basic " + base64.b64encode(b":hunter2").decode()


async def test_create_session_returns_id() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.post("/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_1", "title": "t"})
        )
        async with OpencodeClient(BASE) as client:
            session = await client.create_session(title="t")
    assert session.id == "ses_1"


async def test_opencode_create_session_tags_agent_metadata() -> None:
    """create_session forwards `metadata` in the POST /session body so ownership
    is recorded on the provider side (part 2)."""
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_1"})
        )
        async with OpencodeClient(BASE) as client:
            await client.create_session(metadata={"agent_id": "builder"})
    body = json.loads(route.calls[0].request.content)
    assert body["metadata"] == {"agent_id": "builder"}


async def test_send_message_parses_reply() -> None:
    with respx.mock(base_url=BASE) as mock:
        route = mock.post("/session/ses_1/message").mock(
            return_value=httpx.Response(200, json=_assistant_message())
        )
        async with OpencodeClient(BASE) as client:
            reply = await client.send_message(
                "ses_1",
                SendMessageRequest(
                    parts=[TextPartInput(text="hi")],
                    model=ModelRef(providerID="llamacpp", modelID="gemma-4-26B-A4B-it"),
                    tools={"edit": False},
                ),
            )
    assert isinstance(reply, Message)
    assert reply.text() == "hello from gemma"
    # None fields are omitted; tools + model are sent.
    body = route.calls[0].request.content
    assert b'"agent"' not in body
    assert b'"edit"' in body and b'"providerID"' in body


async def test_send_message_read_timeout_is_unbounded_but_connect_is_capped() -> None:
    """A turn blocks on one synchronous POST the daemon answers only when the model
    is done, so a scalar read timeout would cap the whole turn (the wall-clock bug
    the codex runner had). send_message must disable the READ bound while keeping
    connect/write/pool capped so an unreachable daemon still fails fast."""
    with respx.mock(base_url=BASE) as mock:
        turn = mock.post("/session/ses_1/message").mock(
            return_value=httpx.Response(200, json=_assistant_message())
        )
        create = mock.post("/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_1", "title": "t"})
        )
        async with OpencodeClient(BASE, timeout=30.0) as client:
            await client.create_session()
            await client.send_message(
                "ses_1", SendMessageRequest(parts=[TextPartInput(text="hi")])
            )

    turn_timeout = turn.calls[0].request.extensions["timeout"]
    assert turn_timeout["read"] is None  # the fix: no read cap on a turn
    assert turn_timeout["connect"] == 30.0  # connect still bounded
    # A quick call keeps the bounded read (idle-unbounded is turns-only).
    assert create.calls[0].request.extensions["timeout"]["read"] == 30.0


async def test_send_message_404_is_stale_session() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.post("/session/gone/message").mock(return_value=httpx.Response(404))
        async with OpencodeClient(BASE) as client:
            with pytest.raises(OpencodeStaleSessionError):
                await client.send_message(
                    "gone", SendMessageRequest(parts=[TextPartInput(text="hi")])
                )


async def test_client_and_server_errors_map_by_status() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.post("/session").mock(return_value=httpx.Response(400, json={"e": "bad"}))
        async with OpencodeClient(BASE) as client:
            with pytest.raises(OpencodeClientError):
                await client.create_session()
        mock.post("/session").mock(return_value=httpx.Response(500, text="boom"))
        async with OpencodeClient(BASE) as client:
            with pytest.raises(OpencodeServerError):
                await client.create_session()


async def test_network_error_wrapped() -> None:
    with respx.mock(base_url=BASE) as mock:
        mock.post("/session").mock(side_effect=httpx.ConnectError("nope"))
        async with OpencodeClient(BASE) as client:
            with pytest.raises(OpencodeNetworkError):
                await client.create_session()


async def test_get_messages_parses_list() -> None:
    payload = [
        {"info": {"role": "user"}, "parts": [{"type": "text", "text": "say hi"}]},
        _assistant_message(),
    ]
    with respx.mock(base_url=BASE) as mock:
        mock.get("/session/ses_1/message").mock(
            return_value=httpx.Response(200, json=payload)
        )
        async with OpencodeClient(BASE) as client:
            messages = await client.get_messages("ses_1")
    assert [m.info.role for m in messages] == ["user", "assistant"]
    assert messages[1].info.tokens is not None
    assert messages[1].info.tokens.output == 6
