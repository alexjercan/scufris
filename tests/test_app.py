"""Tests for the FastAPI app: the stats API and static dashboard serving."""

from __future__ import annotations

import json
import logging
import os
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
