"""Tests for the FastAPI app: the stats API and static dashboard serving."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from scufris.agent import (
    AgentReply,
    StreamDone,
    StreamEvent,
    StreamTool,
    TokenUsage,
    ToolCall,
)
from scufris.app import create_app
from scufris.config import McpServerSpec, Settings
from scufris.metrics import Collector
from scufris.processes import ProcessGroup, ProcessInstance, ProcessList


class FakeProcessCollector:
    def sample(self) -> ProcessList:
        return ProcessList(
            groups=[
                ProcessGroup(
                    name="firefox",
                    count=2,
                    cpu_percent=30.0,
                    mem_rss=300,
                    instances=[
                        ProcessInstance(
                            pid=1,
                            username="alex",
                            cpu_percent=20.0,
                            mem_rss=200,
                            num_threads=8,
                            status="running",
                        )
                    ],
                )
            ],
            total=2,
        )


class FakeAgent:
    def __init__(self, session_id: str | None = None) -> None:
        self.messages: list[str] = []
        self.resets = 0
        self._session = session_id
        self.image_paths: list[str] | None = None
        self.image_existed: bool | None = None

    async def chat(self, prompt: str) -> AgentReply:
        self.messages.append(prompt)
        return AgentReply(
            text=f"reply: {prompt}",
            status="completed",
            tool_calls=[
                ToolCall(server="scufris", tool="host_stats", status="completed")
            ],
            usage=TokenUsage(input_tokens=120, output_tokens=8),
        )

    async def chat_stream(
        self, prompt: str, image_paths: list[str] | None = None
    ) -> AsyncIterator[StreamEvent]:
        self.messages.append(prompt)
        self.image_paths = image_paths
        # Record that the decoded image file exists while the turn runs (the
        # endpoint writes it before this and cleans it up after).
        self.image_existed = bool(image_paths and os.path.isfile(image_paths[0]))
        yield StreamTool(
            tool=ToolCall(server="scufris", tool="host_stats", status="completed")
        )
        yield StreamDone(
            reply=AgentReply(text=f"reply: {prompt}", status="completed"),
            session_id="sess-x",
        )

    def reset(self) -> None:
        self.resets += 1

    def current_session_id(self) -> str | None:
        return self._session

    def new_session(self) -> None:
        self._session = None

    def switch_session(self, session_id: str) -> None:
        self._session = session_id

    async def aclose(self) -> None:
        return None


def _write_conversation_rollout(
    home: Path, session_id: str, *, cwd: str, turns: list[tuple[str, str]]
) -> None:
    """Write a rollout with a full (role, text) transcript, for fork tests."""
    day = home / "sessions" / "2026" / "07" / "19"
    day.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, object]] = [
        {
            "type": "session_meta",
            "payload": {
                "session_id": session_id,
                "id": session_id,
                "timestamp": "2026-07-19T14:39:30.556Z",
                "cwd": cwd,
                "originator": "codex_exec",
                "git": {"branch": "main"},
            },
        }
    ]
    for role, text in turns:
        kind = "user_message" if role == "user" else "agent_message"
        events.append({"type": "event_msg", "payload": {"type": kind, "message": text}})
    path = day / f"rollout-2026-07-19T14-39-30-{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _write_session_rollout(
    home: Path, session_id: str, *, cwd: str, used_percent: float = 20.0
) -> None:
    """Write one fake codex rollout so the session endpoints have data to read."""
    day = home / "sessions" / "2026" / "07" / "19"
    day.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "type": "session_meta",
            "payload": {
                "session_id": session_id,
                "id": session_id,
                "timestamp": "2026-07-19T14:39:30.556Z",
                "cwd": cwd,
                "originator": "codex_exec",
                "git": {"branch": "main"},
            },
        },
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "list my tasks"},
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "model_context_window": 258400,
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 40,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "total_tokens": 120,
                    },
                },
                "rate_limits": {
                    "plan_type": "plus",
                    "primary": {
                        "used_percent": used_percent,
                        "window_minutes": 10080,
                        "resets_at": 1785074524,
                    },
                    "secondary": None,
                },
            },
        },
    ]
    path = day / f"rollout-2026-07-19T14-39-30-{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _settings(web_dist: Path, *, agent_enabled: bool = True) -> Settings:
    return Settings(web_dist=web_dist, agent_enabled=agent_enabled)


def test_api_stats_returns_snapshot(fake_collector: Collector, tmp_path: Path) -> None:
    app = create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    client = TestClient(app)

    resp = client.get("/api/stats")
    assert resp.status_code == 200

    body = resp.json()
    assert body["hostname"] == "testbox"
    assert body["mem"]["percent"] == 40.0
    assert body["disks"][0]["mountpoint"] == "/"


def test_api_processes_returns_groups(
    fake_collector: Collector, tmp_path: Path
) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path / "absent"),
        process_collector=FakeProcessCollector(),
    )
    resp = TestClient(app).get("/api/processes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert body["groups"][0]["name"] == "firefox"
    assert body["groups"][0]["instances"][0]["pid"] == 1


def test_api_config_exposes_poll_interval(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent", poll_seconds=5.0, agent_enabled=False
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))

    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["poll_seconds"] == 5.0
    assert body["agent_enabled"] is False


def test_requests_are_logged(
    fake_collector: Collector,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    with caplog.at_level(logging.DEBUG, logger="scufris.app"):
        TestClient(app).get("/api/config")
    assert any("/api/config -> 200" in record.getMessage() for record in caplog.records)


def test_chat_returns_agent_reply(fake_collector: Collector, tmp_path: Path) -> None:
    agent = FakeAgent()
    app = create_app(
        collector=fake_collector, settings=_settings(tmp_path / "absent"), agent=agent
    )
    client = TestClient(app)

    resp = client.post("/api/chat", json={"message": "hello agent"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "reply: hello agent"
    assert agent.messages == ["hello agent"]
    # Per-turn metadata rides along with the reply.
    assert body["tool_calls"][0]["tool"] == "host_stats"
    assert body["usage"]["input_tokens"] == 120


def test_chat_stream_emits_sse_frames(
    fake_collector: Collector, tmp_path: Path
) -> None:
    agent = FakeAgent()
    settings = Settings(web_dist=tmp_path / "absent", agent_enabled=True)
    app = create_app(collector=fake_collector, settings=settings, agent=agent)

    resp = TestClient(app).post("/api/chat/stream", json={"message": "hi"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    # Anti-buffering headers so tokens reach the browser as they are yielded.
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"
    assert resp.headers["x-content-type-options"] == "nosniff"
    body = resp.text
    # A leading comment/padding frame flushes past the browser sniff buffer.
    assert body.startswith(":")
    # A live tool frame, then the done frame carrying the reply.
    assert '"kind":"tool"' in body
    assert '"kind":"done"' in body
    assert "reply: hi" in body
    assert agent.messages == ["hi"]


# A 1x1 transparent PNG, base64.
_PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_chat_stream_passes_an_attached_image_to_the_agent(
    fake_collector: Collector, tmp_path: Path
) -> None:
    agent = FakeAgent()
    settings = Settings(web_dist=tmp_path / "absent", agent_enabled=True)
    app = create_app(collector=fake_collector, settings=settings, agent=agent)

    resp = TestClient(app).post(
        "/api/chat/stream",
        json={
            "message": "what is this?",
            "image": {"data_base64": _PNG_1PX, "mime": "image/png"},
        },
    )
    assert resp.status_code == 200
    assert '"kind":"done"' in resp.text
    # The agent got a real, existing image file path during the turn...
    assert agent.image_paths is not None and len(agent.image_paths) == 1
    assert agent.image_paths[0].endswith(".png")
    assert agent.image_existed is True
    # ...and the temp file is cleaned up afterwards.
    assert not os.path.exists(agent.image_paths[0])


def test_chat_stream_rejects_a_non_image_attachment(
    fake_collector: Collector, tmp_path: Path
) -> None:
    agent = FakeAgent()
    settings = Settings(web_dist=tmp_path / "absent", agent_enabled=True)
    app = create_app(collector=fake_collector, settings=settings, agent=agent)

    resp = TestClient(app).post(
        "/api/chat/stream",
        json={
            "message": "hi",
            "image": {"data_base64": "aGVsbG8=", "mime": "text/plain"},
        },
    )
    assert resp.status_code == 200
    assert '"kind": "error"' in resp.text
    assert "unsupported attachment type" in resp.text
    assert agent.messages == []  # the turn never ran


def test_chat_stream_503_when_disabled(
    fake_collector: Collector, tmp_path: Path
) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path / "absent", agent_enabled=False),
    )
    resp = TestClient(app).post("/api/chat/stream", json={"message": "hi"})
    assert resp.status_code == 503


def test_chat_stream_runs_as_a_supervised_background_job(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The turn runs under the supervisor (decoupled from the request): after the
    stream call, the run is tracked and terminal - proving the endpoint relays a
    supervised run rather than iterating the agent inline (ADR-001)."""
    agent = FakeAgent()
    settings = Settings(web_dist=tmp_path / "absent", agent_enabled=True)
    app = create_app(collector=fake_collector, settings=settings, agent=agent)

    resp = TestClient(app).post("/api/chat/stream", json={"message": "hi"})
    assert resp.status_code == 200

    runs = app.state.supervisor.list_runs()
    assert len(runs) == 1
    assert runs[0].state == "done"


def test_static_bundle_served_with_no_cache(
    fake_collector: Collector, tmp_path: Path
) -> None:
    # The non-hashed SPA bundle must revalidate on every load so a rebuild is
    # picked up immediately instead of a stale copy running for hours.
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "agent.js").write_text("console.log('v1');")
    app = create_app(collector=fake_collector, settings=Settings(web_dist=dist))
    resp = TestClient(app).get("/agent.js")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-cache"


def test_agent_info_reports_model_and_state(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_model="gpt-5.5",
        agent_enabled=True,
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    body = client.get("/api/agent/info").json()
    assert body["model"] == "gpt-5.5"
    assert body["enabled"] is True
    assert body["auth_mode"] == "chatgpt"


def test_agent_tools_lists_the_mcp_tools(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    )
    body = client.get("/api/agent/tools").json()
    names = {t["name"] for t in body}
    assert {"host_stats", "tatr_ls", "tatr_show"} <= names
    assert all(t["description"] for t in body)
    # Each tool reports its source server and its argument names (from the schema).
    assert all(t["server"] == "scufris" for t in body)
    tatr_ls = next(t for t in body if t["name"] == "tatr_ls")
    assert set(tatr_ls["args"]) == {"filter", "sort"}


def test_agent_health_endpoint_reports_checks(
    fake_collector: Collector, tmp_path: Path
) -> None:
    # A fake codex_bin keeps the probe deterministic (no real codex subprocess).
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_backend="mock",
        agent_tools_enabled=True,
        codex_bin=str(tmp_path / "no-such-codex"),
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    body = client.get("/api/agent/health").json()
    assert body["scufris_version"]
    by_name = {c["name"]: c["status"] for c in body["checks"]}
    assert by_name["agent"] == "ok"
    assert by_name["mcp tools"] == "ok"
    assert by_name["web assets"] == "error"


def test_agent_config_reports_effective_settings(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_backend="app_server",
        agent_model="gpt-5.5",
        agent_tools_enabled=True,
        mcp_servers=[McpServerSpec(id="extra", command="mcp-extra")],
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    body = client.get("/api/agent/config").json()
    assert body["backend"] == "app_server"
    assert body["model"] == "gpt-5.5"
    assert body["auth_mode"] == "chatgpt"
    assert body["sandbox"] == "read-only"
    assert body["tools_enabled"] is True
    servers = {s["id"]: s["source"] for s in body["mcp_servers"]}
    assert servers == {"scufris": "built-in", "extra": "configured"}


def test_agent_config_omits_builtin_server_when_tools_disabled(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_tools_enabled=False,
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    body = client.get("/api/agent/config").json()
    assert body["tools_enabled"] is False
    assert body["mcp_servers"] == []


def test_agent_config_reports_writable(
    fake_collector: Collector, tmp_path: Path
) -> None:
    writable = Settings(web_dist=tmp_path / "absent", state_dir=tmp_path / "st")
    body = TestClient(
        create_app(collector=fake_collector, settings=writable, agent=FakeAgent())
    ).get("/api/agent/config")
    assert body.json()["writable"] is True
    ro = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path / "st", settings_writable=False
    )
    body = TestClient(
        create_app(collector=fake_collector, settings=ro, agent=FakeAgent())
    ).get("/api/agent/config")
    assert body.json()["writable"] is False


def test_patch_agent_config_persists(fake_collector: Collector, tmp_path: Path) -> None:
    # A non-injected agent so the store's handle path is exercised; mock backend
    # needs no codex. The change must apply and survive into a fresh app.
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend="mock",
        agent_model="gpt-5.5",
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    resp = client.patch(
        "/api/agent/config",
        json={"agent_model": "gpt-5.6", "agent_tools_enabled": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "gpt-5.6"
    assert body["tools_enabled"] is False
    assert (tmp_path / "settings.json").is_file()

    fresh = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend="mock"
    )
    body2 = (
        TestClient(create_app(collector=fake_collector, settings=fresh))
        .get("/api/agent/config")
        .json()
    )
    assert body2["model"] == "gpt-5.6"
    assert body2["tools_enabled"] is False


def test_patch_agent_config_rebuilds_on_backend_change(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend="mock"
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    resp = client.patch("/api/agent/config", json={"agent_backend": "exec"})
    assert resp.status_code == 200
    assert resp.json()["backend"] == "exec"


def test_patch_agent_config_forbidden_when_readonly(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        settings_writable=False,
        agent_backend="mock",
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    resp = client.patch("/api/agent/config", json={"agent_model": "gpt-5.6"})
    assert resp.status_code == 403
    assert not (tmp_path / "settings.json").exists()


def test_patch_agent_config_rejects_non_whitelisted(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend="mock"
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    resp = client.patch("/api/agent/config", json={"openai_api_key": "sk-secret"})
    assert resp.status_code == 422


def test_tools_endpoint_reports_enabled(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        disabled_tools=["tatr_new"],
    )
    client = TestClient(
        create_app(collector=fake_collector, settings=settings, agent=FakeAgent())
    )
    tools = {t["name"]: t["enabled"] for t in client.get("/api/agent/tools").json()}
    assert tools["tatr_new"] is False
    assert tools["host_stats"] is True


def test_patch_disabled_tools_persists(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend="mock"
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    resp = client.patch("/api/agent/config", json={"disabled_tools": ["disk_usage"]})
    assert resp.status_code == 200
    # A fresh app over the same state dir marks it disabled in the tools list.
    fresh = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend="mock"
    )
    tools = {
        t["name"]: t["enabled"]
        for t in TestClient(create_app(collector=fake_collector, settings=fresh))
        .get("/api/agent/tools")
        .json()
    }
    assert tools["disk_usage"] is False


def test_add_mcp_server_persists(fake_collector: Collector, tmp_path: Path) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend="mock"
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    resp = client.patch(
        "/api/agent/config",
        json={"mcp_servers": [{"id": "fs", "command": "mcp-fs"}]},
    )
    assert resp.status_code == 200
    ids = {s["id"] for s in resp.json()["mcp_servers"]}
    assert "fs" in ids
    fresh = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend="mock"
    )
    ids2 = {
        s["id"]
        for s in TestClient(create_app(collector=fake_collector, settings=fresh))
        .get("/api/agent/config")
        .json()["mcp_servers"]
    }
    assert "fs" in ids2


@pytest.mark.parametrize(
    "server",
    [
        {"id": "bad id", "command": "x"},  # space
        {"id": "fs\n", "command": "x"},  # trailing newline (fullmatch, not $)
        {"id": "fs.sub", "command": "x"},  # dot is not a bare TOML key
        {"id": "scufris", "command": "x"},  # reserved built-in id
        {"id": "fs", "command": "   "},  # empty/whitespace command
    ],
)
def test_add_mcp_server_rejects_bad_id(
    fake_collector: Collector, tmp_path: Path, server: dict[str, str]
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend="mock"
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    resp = client.patch("/api/agent/config", json={"mcp_servers": [server]})
    assert resp.status_code == 422
    assert not (tmp_path / "settings.json").exists()  # nothing persisted


def test_post_mcp_server_appends_and_persists(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    resp = client.post("/api/agent/mcp_servers", json={"id": "fs", "command": "mcp-fs"})
    assert resp.status_code == 200
    assert "fs" in {s["id"] for s in resp.json()["mcp_servers"]}
    # A second, different server appends (does not replace the first).
    resp2 = client.post(
        "/api/agent/mcp_servers", json={"id": "gh", "command": "mcp-gh"}
    )
    assert {"scufris", "fs", "gh"} == {s["id"] for s in resp2.json()["mcp_servers"]}
    # Persisted: a fresh app over the same state dir still has them.
    fresh = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    ids = {s["id"] for s in fresh.get("/api/agent/config").json()["mcp_servers"]}
    assert {"fs", "gh"} <= ids


def test_post_mcp_server_rejects_duplicate(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    client.post("/api/agent/mcp_servers", json={"id": "fs", "command": "mcp-fs"})
    dup = client.post("/api/agent/mcp_servers", json={"id": "fs", "command": "other"})
    assert dup.status_code == 409


@pytest.mark.parametrize(
    "server", [{"id": "bad id", "command": "x"}, {"id": "scufris", "command": "x"}]
)
def test_post_mcp_server_rejects_bad_or_reserved_id(
    fake_collector: Collector, tmp_path: Path, server: dict[str, str]
) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    assert client.post("/api/agent/mcp_servers", json=server).status_code == 422


def test_delete_mcp_server_removes_and_404s_unknown(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    client.post("/api/agent/mcp_servers", json={"id": "fs", "command": "mcp-fs"})
    ok = client.delete("/api/agent/mcp_servers/fs")
    assert ok.status_code == 200
    assert "fs" not in {s["id"] for s in ok.json()["mcp_servers"]}
    assert client.delete("/api/agent/mcp_servers/ghost").status_code == 404


def test_mcp_server_endpoints_forbidden_when_readonly(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend="mock",
        settings_writable=False,
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    assert (
        client.post(
            "/api/agent/mcp_servers", json={"id": "fs", "command": "mcp-fs"}
        ).status_code
        == 403
    )
    # A read-only server has no configured servers to delete, but the gate must
    # trip before the 404: seed one via env so there IS a target.
    seeded = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path / "ro2",
        agent_backend="mock",
        settings_writable=False,
        mcp_servers=[McpServerSpec(id="fs", command="mcp-fs")],
    )
    ro = TestClient(create_app(collector=fake_collector, settings=seeded))
    assert ro.delete("/api/agent/mcp_servers/fs").status_code == 403


def _mock_settings(tmp_path: Path) -> Settings:
    return Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend="mock",
        enable_mock_backend=True,  # allow creating mock-backed agent records
    )


def test_profiles_list_create_activate_flow(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    client.patch("/api/agent/config", json={"agent_model": "gpt-5.5"})
    assert client.get("/api/agent/profiles").json() == {
        "profiles": ["default"],
        "active": "default",
    }

    resp = client.post("/api/agent/profiles", json={"name": "cheap"})
    assert resp.status_code == 200
    assert set(resp.json()["profiles"]) == {"default", "cheap"}

    assert (
        client.post("/api/agent/profiles/activate", json={"name": "cheap"}).status_code
        == 200
    )
    client.patch("/api/agent/config", json={"agent_model": "gpt-5-mini"})
    assert client.get("/api/agent/config").json()["model"] == "gpt-5-mini"

    back = client.post("/api/agent/profiles/activate", json={"name": "default"})
    assert back.json()["model"] == "gpt-5.5"


def test_profile_create_rejects_bad_name(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    assert (
        client.post("/api/agent/profiles", json={"name": "has space"}).status_code
        == 422
    )
    assert (
        client.post("/api/agent/profiles", json={"name": "default"}).status_code == 409
    )


def test_profile_delete_and_guards(fake_collector: Collector, tmp_path: Path) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    client.post("/api/agent/profiles", json={"name": "temp"})
    assert client.delete("/api/agent/profiles/default").status_code == 409  # active
    ok = client.delete("/api/agent/profiles/temp")
    assert ok.status_code == 200
    assert ok.json()["profiles"] == ["default"]
    assert client.delete("/api/agent/profiles/ghost").status_code == 404


def test_profile_write_forbidden_when_readonly(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend="mock",
        settings_writable=False,
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    assert client.post("/api/agent/profiles", json={"name": "x"}).status_code == 403


def test_projects_crud_endpoints(fake_collector: Collector, tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    assert client.get("/api/projects").json() == []

    created = client.post(
        "/api/projects",
        json={"name": "My App", "cwd": str(proj), "language": "python"},
    )
    assert created.status_code == 200
    pid = created.json()["id"]
    assert pid == "my-app"

    assert client.get(f"/api/projects/{pid}").json()["name"] == "My App"
    assert client.get("/api/projects/ghost").status_code == 404

    patched = client.patch(f"/api/projects/{pid}", json={"description": "updated"})
    assert patched.status_code == 200
    assert patched.json()["description"] == "updated"

    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert client.get("/api/projects").json() == []
    assert client.delete("/api/projects/ghost").status_code == 404


def test_project_create_validation(fake_collector: Collector, tmp_path: Path) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    # Missing cwd dir -> 422.
    bad = client.post(
        "/api/projects", json={"name": "x", "cwd": str(tmp_path / "nope")}
    )
    assert bad.status_code == 422
    # Empty name -> 422.
    proj = tmp_path / "proj"
    proj.mkdir()
    assert (
        client.post("/api/projects", json={"name": "  ", "cwd": str(proj)}).status_code
        == 422
    )


def test_projects_write_forbidden_when_readonly(
    fake_collector: Collector, tmp_path: Path
) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend="mock",
        settings_writable=False,
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    resp = client.post("/api/projects", json={"name": "x", "cwd": str(proj)})
    assert resp.status_code == 403
    # PATCH and DELETE must be gated too - the read-only gate trips before the
    # store's 404, so a nonexistent id still returns 403.
    assert client.patch("/api/projects/any", json={"name": "y"}).status_code == 403
    assert client.delete("/api/projects/any").status_code == 403


@pytest.mark.needs_tatr
def test_project_tasks_endpoint(fake_collector: Collector, tmp_path: Path) -> None:
    import subprocess

    proj = tmp_path / "proj"
    (proj / "tasks").mkdir(parents=True)  # tatr needs an existing tasks/ dir
    subprocess.run(
        ["tatr", "-r", str(proj), "new", "spec one", "-p", "20", "-t", "feature"],
        check=True,
        capture_output=True,
        text=True,
    )
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    created = client.post("/api/projects", json={"name": "P", "cwd": str(proj)}).json()
    body = client.get(f"/api/projects/{created['id']}/tasks").json()
    assert len(body) == 1
    assert body[0]["title"] == "spec one"
    assert body[0]["priority"] == 20
    assert body[0]["tags"] == ["feature"]


def test_project_tasks_empty_when_no_tasks_dir(
    fake_collector: Collector, tmp_path: Path
) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()  # no tasks/ inside
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    created = client.post("/api/projects", json={"name": "P", "cwd": str(proj)}).json()
    assert client.get(f"/api/projects/{created['id']}/tasks").json() == []


def test_project_tasks_unknown_404(fake_collector: Collector, tmp_path: Path) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    assert client.get("/api/projects/ghost/tasks").status_code == 404


def test_chat_reset_resets_agent(fake_collector: Collector, tmp_path: Path) -> None:
    agent = FakeAgent()
    app = create_app(
        collector=fake_collector, settings=_settings(tmp_path / "absent"), agent=agent
    )
    client = TestClient(app)

    resp = client.post("/api/chat/reset")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert agent.resets == 1


def _agent_settings(web_dist: Path, codex_home: Path) -> Settings:
    return Settings(web_dist=web_dist, agent_enabled=True, codex_home=codex_home)


def test_sessions_lists_and_reports_current(
    fake_collector: Collector, tmp_path: Path
) -> None:
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-1", cwd=os.getcwd())
    agent = FakeAgent(session_id="sess-1")
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
        agent=agent,
    )
    body = TestClient(app).get("/api/agent/sessions").json()
    assert body["current"] == "sess-1"
    assert [s["id"] for s in body["sessions"]] == ["sess-1"]
    assert body["sessions"][0]["title"] == "list my tasks"


def test_sessions_empty_when_disabled(
    fake_collector: Collector, tmp_path: Path
) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path / "absent", agent_enabled=False),
    )
    body = TestClient(app).get("/api/agent/sessions").json()
    assert body == {"sessions": [], "current": None}


def test_session_switch_and_new(fake_collector: Collector, tmp_path: Path) -> None:
    agent = FakeAgent()
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", tmp_path / "codex"),
        agent=agent,
    )
    client = TestClient(app)

    switched = client.post(
        "/api/agent/session", json={"action": "switch", "session_id": "sess-9"}
    )
    assert switched.status_code == 200
    assert switched.json()["current"] == "sess-9"
    assert agent.current_session_id() == "sess-9"

    fresh = client.post("/api/agent/session", json={"action": "new"})
    assert fresh.json()["current"] is None


def test_session_switch_requires_id(fake_collector: Collector, tmp_path: Path) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", tmp_path / "codex"),
        agent=FakeAgent(),
    )
    resp = TestClient(app).post("/api/agent/session", json={"action": "switch"})
    assert resp.status_code == 422


def test_session_post_503_when_disabled(
    fake_collector: Collector, tmp_path: Path
) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path / "absent", agent_enabled=False),
    )
    resp = TestClient(app).post("/api/agent/session", json={"action": "new"})
    assert resp.status_code == 503


def test_context_endpoint_returns_snapshot(
    fake_collector: Collector, tmp_path: Path
) -> None:
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-ctx", cwd=os.getcwd())
    agent = FakeAgent(session_id="sess-ctx")
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
        agent=agent,
    )
    body = TestClient(app).get("/api/agent/context").json()
    assert body["session_id"] == "sess-ctx"
    assert body["context_window"] == 258400
    assert body["input_tokens"] == 100
    assert body["turn_count"] == 1


def test_context_null_when_no_current_session(
    fake_collector: Collector, tmp_path: Path
) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", tmp_path / "codex"),
        agent=FakeAgent(session_id=None),
    )
    assert TestClient(app).get("/api/agent/context").json() is None


def test_session_transcript_returns_messages(
    fake_collector: Collector, tmp_path: Path
) -> None:
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-t", cwd=os.getcwd())
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
        agent=FakeAgent(),
    )
    body = TestClient(app).get("/api/agent/session/sess-t").json()
    first = body["messages"][0]
    assert first["role"] == "user"
    assert first["text"] == "list my tasks"
    assert "ts" in first  # timestamp field present (None when the event had none)


def test_session_transcript_empty_when_disabled(
    fake_collector: Collector, tmp_path: Path
) -> None:
    app = create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    body = TestClient(app).get("/api/agent/session/whatever").json()
    assert body == {"messages": []}


def test_delete_session_removes_and_resets_current(
    fake_collector: Collector, tmp_path: Path
) -> None:
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-del", cwd=os.getcwd())
    agent = FakeAgent(session_id="sess-del")
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
        agent=agent,
    )
    client = TestClient(app)

    resp = client.delete("/api/agent/session/sess-del")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] is True
    assert body["current"] is None  # was the active session -> reset
    # It is gone from the list.
    listed = client.get("/api/agent/sessions").json()["sessions"]
    assert listed == []


def test_delete_session_keeps_current_when_other(
    fake_collector: Collector, tmp_path: Path
) -> None:
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-a", cwd=os.getcwd())
    agent = FakeAgent(session_id="sess-current")
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
        agent=agent,
    )
    body = TestClient(app).delete("/api/agent/session/sess-a").json()
    assert body["deleted"] is True
    assert body["current"] == "sess-current"  # a different session stays active


def test_delete_session_503_when_disabled(
    fake_collector: Collector, tmp_path: Path
) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path / "absent", agent_enabled=False),
    )
    resp = TestClient(app).delete("/api/agent/session/whatever")
    assert resp.status_code == 503


def test_fork_seeds_new_session_with_prior_context(
    fake_collector: Collector, tmp_path: Path
) -> None:
    home = tmp_path / "codex"
    _write_conversation_rollout(
        home,
        "sess-src",
        cwd=os.getcwd(),
        turns=[
            ("user", "first question"),
            ("assistant", "first answer"),
            ("user", "second question"),
        ],
    )
    agent = FakeAgent(session_id="sess-src")
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
        agent=agent,
    )
    # Fork at the second user message (index 2), editing its text.
    resp = TestClient(app).post(
        "/api/agent/session/fork",
        json={"source_id": "sess-src", "message_index": 2, "text": "edited second"},
    )
    assert resp.status_code == 200
    # The seed prompt (what the agent was asked) carries the prior turns + the edit.
    seed = agent.messages[-1]
    assert "first question" in seed
    assert "first answer" in seed
    assert seed.rstrip().endswith("edited second")
    # The message AFTER the fork point (the original "second question") is dropped.
    assert "second question" not in seed


def test_fork_503_when_disabled(fake_collector: Collector, tmp_path: Path) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path / "absent", agent_enabled=False),
    )
    resp = TestClient(app).post(
        "/api/agent/session/fork",
        json={"source_id": "x", "message_index": 0, "text": "hi"},
    )
    assert resp.status_code == 503


def test_usage_endpoint_returns_weekly_window(
    fake_collector: Collector, tmp_path: Path
) -> None:
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-u", cwd=os.getcwd(), used_percent=42.0)
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
        agent=FakeAgent(),
    )
    body = TestClient(app).get("/api/agent/usage").json()
    assert body["plan_type"] == "plus"
    assert body["primary"]["window_minutes"] == 10080
    assert body["primary"]["used_percent"] == 42.0


def test_usage_null_when_disabled(fake_collector: Collector, tmp_path: Path) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path / "absent", agent_enabled=False),
    )
    assert TestClient(app).get("/api/agent/usage").json() is None


def test_memory_endpoint_reports_footprint(
    fake_collector: Collector, tmp_path: Path
) -> None:
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-a", cwd=os.getcwd())
    _write_session_rollout(home, "sess-b", cwd=os.getcwd())
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
        agent=FakeAgent(),
    )
    body = TestClient(app).get("/api/agent/memory").json()
    assert body["session_count"] == 2
    assert body["total_bytes"] > 0
    assert body["oldest"] is not None and body["newest"] is not None


def test_memory_endpoint_empty_ok(fake_collector: Collector, tmp_path: Path) -> None:
    # Missing sessions dir -> zeros, not an error.
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", tmp_path / "no-codex"),
        agent=FakeAgent(),
    )
    body = TestClient(app).get("/api/agent/memory").json()
    assert body == {
        "session_count": 0,
        "total_bytes": 0,
        "oldest": None,
        "newest": None,
    }


def test_memory_zero_when_disabled(fake_collector: Collector, tmp_path: Path) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path / "absent", agent_enabled=False),
    )
    assert TestClient(app).get("/api/agent/memory").json()["session_count"] == 0


def test_account_endpoint_shape(fake_collector: Collector, tmp_path: Path) -> None:
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-acc", cwd=os.getcwd(), used_percent=17.0)
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
        agent=FakeAgent(),
    )
    body = TestClient(app).get("/api/agent/account").json()
    assert body["auth_mode"] == "chatgpt"
    assert body["model"]  # non-empty
    assert body["enabled"] is True
    assert body["quota"]["primary"]["used_percent"] == 17.0


def test_account_quota_null_when_disabled(
    fake_collector: Collector, tmp_path: Path
) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path / "absent", agent_enabled=False),
    )
    body = TestClient(app).get("/api/agent/account").json()
    assert body["enabled"] is False
    assert body["quota"] is None


def test_chat_returns_503_when_agent_disabled(
    fake_collector: Collector, tmp_path: Path
) -> None:
    # Explicitly disabled agent, so chat is unavailable.
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path / "absent", agent_enabled=False),
    )
    client = TestClient(app)

    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 503
    assert "disabled" in resp.json()["detail"]


def test_index_served_when_dist_exists(
    fake_collector: Collector, tmp_path: Path
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>scufris</html>")

    app = create_app(collector=fake_collector, settings=_settings(dist))
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "scufris" in resp.text


def test_stats_page_served_at_subpath(
    fake_collector: Collector, tmp_path: Path
) -> None:
    dist = tmp_path / "dist"
    (dist / "stats").mkdir(parents=True)
    (dist / "index.html").write_text("<html>agent</html>")
    (dist / "stats" / "index.html").write_text("<html>stats page</html>")

    app = create_app(collector=fake_collector, settings=_settings(dist))
    client = TestClient(app)

    resp = client.get("/stats/")
    assert resp.status_code == 200
    assert "stats page" in resp.text


def test_api_wins_over_static_mount(fake_collector: Collector, tmp_path: Path) -> None:
    # Even with the static bundle mounted at "/", the API route must resolve.
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>scufris</html>")

    app = create_app(collector=fake_collector, settings=_settings(dist))
    client = TestClient(app)

    resp = client.get("/api/stats")
    assert resp.status_code == 200
    assert resp.json()["hostname"] == "testbox"


def _client_with_project(fake_collector: Collector, tmp_path: Path) -> TestClient:
    """A writable app with a single project 'my-app' so agents can bind to it."""
    proj = tmp_path / "proj"
    proj.mkdir()
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    client.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
    return client


def test_agents_crud_endpoints(fake_collector: Collector, tmp_path: Path) -> None:
    client = _client_with_project(fake_collector, tmp_path)
    assert client.get("/api/agents").json() == []

    created = client.post(
        "/api/agents",
        json={
            "name": "Builder",
            "project_id": "my-app",
            "backend": "mock",
            "goal": "do the thing",
        },
    )
    assert created.status_code == 200
    body = created.json()
    assert body["id"] == "builder"
    assert body["project_id"] == "my-app"
    assert body["state"] == "idle"
    assert body["permission_mode"] == "manual"

    assert client.get("/api/agents/builder").json()["name"] == "Builder"
    assert client.get("/api/agents/ghost").status_code == 404

    patched = client.patch("/api/agents/builder", json={"permission_mode": "edit"})
    assert patched.status_code == 200
    assert patched.json()["permission_mode"] == "edit"

    assert client.delete("/api/agents/builder").status_code == 200
    assert client.get("/api/agents").json() == []
    assert client.delete("/api/agents/ghost").status_code == 404


def test_agent_create_validation(fake_collector: Collector, tmp_path: Path) -> None:
    client = _client_with_project(fake_collector, tmp_path)
    # Unknown project -> 422.
    assert (
        client.post("/api/agents", json={"name": "x", "project_id": "nope"}).status_code
        == 422
    )
    # Bad backend -> 422.
    assert (
        client.post(
            "/api/agents",
            json={"name": "x", "project_id": "my-app", "backend": "zzz"},
        ).status_code
        == 422
    )
    # Patch of an unknown agent -> 404.
    assert client.patch("/api/agents/ghost", json={"name": "y"}).status_code == 404
    # Unknown field on patch -> 422 (AgentUpdate forbids extras).
    client.post(
        "/api/agents",
        json={"name": "A", "project_id": "my-app", "backend": "mock"},
    )
    assert client.patch("/api/agents/a", json={"nonsense": 1}).status_code == 422
    # project_id is immutable via PATCH (not an AgentUpdate field) -> 422.
    assert (
        client.patch("/api/agents/a", json={"project_id": "other"}).status_code == 422
    )


def test_agents_write_forbidden_when_readonly(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend="mock",
        settings_writable=False,
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    # The read-only gate trips before the store's 404/422.
    assert (
        client.post("/api/agents", json={"name": "x", "project_id": "any"}).status_code
        == 403
    )
    assert client.patch("/api/agents/any", json={"name": "y"}).status_code == 403
    assert client.delete("/api/agents/any").status_code == 403


def _agent_client(
    fake_collector: Collector, tmp_path: Path, *, goal: str = "do the thing"
) -> TestClient:
    """A mock-backend app with project 'my-app' and agent 'builder' (a goal)."""
    proj = tmp_path / "proj"
    proj.mkdir()
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    client.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
    client.post(
        "/api/agents",
        json={
            "name": "Builder",
            "project_id": "my-app",
            "backend": "mock",
            "goal": goal,
        },
    )
    return client


def _wait_state(
    client: TestClient, agent_id: str, target: str, tries: int = 200
) -> dict:
    """Poll status until the background run reaches `target` (the portal loop runs
    the run between our polls)."""
    st: dict = {}
    for _ in range(tries):
        st = client.get(f"/api/agents/{agent_id}/status").json()
        if st.get("state") == target:
            return st
        time.sleep(0.01)
    return st


def test_agent_run_reaches_done_and_persists_session(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = _agent_client(fake_collector, tmp_path)
    started = client.post("/api/agents/builder/run", json={})
    assert started.status_code == 200
    # The launch reports the supervisor's real state (queued until a slot frees).
    assert started.json()["state"] in ("queued", "running")

    st = _wait_state(client, "builder", "done")
    assert st["state"] == "done"
    # The mock run produced a session id, now persisted on the agent.
    agent = client.get("/api/agents/builder").json()
    assert agent["session_id"] == "mock-session"
    assert agent["state"] == "done"
    # Status merges the backend read_status (mock reports turns=1).
    assert st["turns"] == 1
    assert st["last_message"] == "[mock] running"


def test_agent_run_requires_a_goal(fake_collector: Collector, tmp_path: Path) -> None:
    client = _agent_client(fake_collector, tmp_path)
    client.post(
        "/api/agents",
        json={"name": "NoGoal", "project_id": "my-app", "backend": "mock"},
    )
    # No stored goal and no override -> 422.
    assert client.post("/api/agents/nogoal/run", json={}).status_code == 422
    # An override goal runs.
    assert client.post("/api/agents/nogoal/run", json={"goal": "x"}).status_code == 200
    # Unknown agent -> 404.
    assert client.post("/api/agents/ghost/run", json={}).status_code == 404


def test_agent_can_be_rerun_after_completion(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A finished prior run (unique run id per launch) does not block a re-run."""
    client = _agent_client(fake_collector, tmp_path)
    assert client.post("/api/agents/builder/run", json={}).status_code == 200
    _wait_state(client, "builder", "done")
    assert client.post("/api/agents/builder/run", json={}).status_code == 200


def test_agent_events_relay(fake_collector: Collector, tmp_path: Path) -> None:
    client = _agent_client(fake_collector, tmp_path)
    # No run yet -> 404 on events.
    assert client.get("/api/agents/builder/events").status_code == 404

    client.post("/api/agents/builder/run", json={})
    _wait_state(client, "builder", "done")
    # The run's bus (buffered) replays through the SSE relay.
    resp = client.get("/api/agents/builder/events")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert '"kind":"done"' in resp.text


def test_openapi_docs_are_organized(fake_collector: Collector, tmp_path: Path) -> None:
    app = create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    schema = app.openapi()

    # Info metadata is filled in (title, version, a description).
    assert schema["info"]["title"] == "Scufris API"
    assert schema["info"]["version"]
    assert "orchestrator" in schema["info"]["description"].lower()

    # Tag sections are present, in the intended order, each with a description.
    tag_names = [t["name"] for t in schema["tags"]]
    assert tag_names == [
        "host",
        "app",
        "chat",
        "sessions",
        "settings",
        "projects",
        "agents",
    ]
    assert all(t.get("description") for t in schema["tags"])

    def tag_of(path: str, method: str = "get") -> list[str]:
        return schema["paths"][path][method].get("tags", [])

    assert tag_of("/api/stats") == ["host"]
    assert tag_of("/api/config") == ["app"]
    assert tag_of("/api/chat/stream", "post") == ["chat"]
    assert tag_of("/api/agent/info") == ["chat"]  # chat, not settings
    assert tag_of("/api/agent/sessions") == ["sessions"]
    assert tag_of("/api/agent/config") == ["settings"]
    assert tag_of("/api/projects", "post") == ["projects"]
    assert tag_of("/api/agents", "get") == ["agents"]  # plural, not settings
    assert tag_of("/api/agents/{agent_id}/run", "post") == ["agents"]

    # Every API operation is tagged (no orphan in an "default" section).
    for path, ops in schema["paths"].items():
        if not path.startswith("/api/"):
            continue
        for method, op in ops.items():
            assert op.get("tags"), f"{method.upper()} {path} has no OpenAPI tag"

    # /docs and the schema itself serve.
    client = TestClient(app)
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_agent_detail_page_serves_shell(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """/agents/<id> serves the agent-detail SPA shell; /api/agents/<id> is
    unaffected; /agents/ (list) is not shadowed by the detail route."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "agent-detail.html").write_text("<html>DETAIL SHELL</html>")
    (dist / "index.html").write_text("<html>landing</html>")
    (dist / "agents").mkdir()
    (dist / "agents" / "index.html").write_text("<html>AGENTS LIST</html>")
    settings = Settings(
        web_dist=dist,
        state_dir=tmp_path,
        agent_backend="mock",
        enable_mock_backend=True,
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))

    # A specific agent path -> the detail shell.
    detail = client.get("/agents/builder")
    assert detail.status_code == 200
    assert "DETAIL SHELL" in detail.text
    # A sub-path (e.g. settings) -> the same shell.
    assert "DETAIL SHELL" in client.get("/agents/builder/settings").text
    # The list path -> the static agents index, NOT the detail shell.
    assert "AGENTS LIST" in client.get("/agents/").text
    # The JSON API for an agent is unaffected (404 for an unknown id, not the shell).
    api = client.get("/api/agents/builder")
    assert api.status_code == 404
    assert "DETAIL SHELL" not in api.text


def test_agent_detail_page_404_without_frontend(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(web_dist=tmp_path / "absent", state_dir=tmp_path)
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    assert client.get("/agents/builder").status_code == 404
