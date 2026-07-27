"""Tests for the `agent` MCP server (scufris.agent_mcp_server): the sub-agent
callback tools request_input + report_back.

The tools POST to a respx-stubbed local dashboard API, so the HTTP plumbing is
exercised without a live server. Tools are called directly (FastMCP's decorator
returns the original function).
"""

from __future__ import annotations

import json

import httpx
import respx

from scufris.agent_mcp_server import mcp, report_back, request_input

_BASE = "http://127.0.0.1:8000"


async def test_agent_server_exposes_only_the_callbacks() -> None:
    names = {tool.name for tool in await mcp.list_tools()}
    assert names == {"request_input", "report_back"}
    assert all(tool.description for tool in await mcp.list_tools())


@respx.mock
def test_request_input_posts_question_for_the_env_agent(monkeypatch) -> None:
    """request_input addresses the caller's own id (SCUFRIS_AGENT_ID) and posts
    the question to the request_input endpoint."""
    monkeypatch.setenv("SCUFRIS_AGENT_ID", "builder")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"agent_id": "builder", "state": "waiting"})

    route = respx.post(f"{_BASE}/api/agents/builder/request_input").mock(
        side_effect=handler
    )
    out = request_input("should I merge to master?")
    assert route.called
    assert seen["body"] == {"question": "should I merge to master?"}
    assert "waiting" in out


def test_request_input_without_agent_id_is_an_error(monkeypatch) -> None:
    """With no SCUFRIS_AGENT_ID in the environment, request_input refuses rather
    than posting to a bogus path."""
    monkeypatch.delenv("SCUFRIS_AGENT_ID", raising=False)
    out = request_input("merge?")
    assert out.startswith("error:")


def test_request_input_requires_a_question(monkeypatch) -> None:
    monkeypatch.setenv("SCUFRIS_AGENT_ID", "builder")
    assert request_input("   ").startswith("error:")


@respx.mock
def test_report_back_posts_summary_for_the_env_agent(monkeypatch) -> None:
    """report_back addresses the caller's own id (SCUFRIS_AGENT_ID) and posts the
    summary to the report_back endpoint."""
    monkeypatch.setenv("SCUFRIS_AGENT_ID", "builder")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"agent_id": "builder", "state": "reported"})

    route = respx.post(f"{_BASE}/api/agents/builder/report_back").mock(
        side_effect=handler
    )
    out = report_back("implemented X; tests green")
    assert route.called
    assert seen["body"] == {"summary": "implemented X; tests green"}
    assert "reported" in out


def test_report_back_without_agent_id_is_an_error(monkeypatch) -> None:
    """With no SCUFRIS_AGENT_ID in the environment, report_back refuses rather than
    posting to a bogus path."""
    monkeypatch.delenv("SCUFRIS_AGENT_ID", raising=False)
    out = report_back("done")
    assert out.startswith("error:")


def test_report_back_requires_a_summary(monkeypatch) -> None:
    monkeypatch.setenv("SCUFRIS_AGENT_ID", "builder")
    assert report_back("   ").startswith("error:")
