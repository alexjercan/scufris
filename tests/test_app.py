"""Tests for the FastAPI app: the stats API and static dashboard serving."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import AsyncIterator

import pytest
from conftest import patch_get_backend
from fastapi.testclient import TestClient
from sqlalchemy import text as sql_text

from scufris.agent import (
    AgentReply,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamReasoningDelta,
    StreamSessionStarted,
    StreamTextDelta,
    StreamTool,
    TokenUsage,
    ToolCall,
)
from scufris.agent_store import AgentRecord, AgentStore
from scufris.app import create_app
from scufris.config import Settings
from scufris.db import Database, state_database
from scufris.enums import AgentState, AuthMode, Backend
from scufris.env_bridge import ensure_api_base, ensure_den_path
from scufris.projects import Project, ProjectStore
from scufris.reasoning_store import ReasoningStore
from scufris.sessions import STEERING_PREAMBLE, TranscriptMessage
from scufris_host import (
    Collector,
    HostOverview,
    HostStats,
    ProcessGroup,
    ProcessInstance,
    ProcessList,
)


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


class FakeBackend:
    """A rich in-test backend for the landing orchestrator chat.

    Injected by ``tests/conftest.py::patch_get_backend`` so the orchestrator's
    turns run through it (the landing chat no longer has an injected agent - it
    goes through ``get_backend(agent.backend).stream()`` like any agent). It
    records the prompts + attached image it saw and emits a live tool frame plus
    a done frame carrying tool_calls/usage, so the endpoint's metadata relay and
    image passthrough stay observable.
    """

    name = "fake"

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.image_paths: list[str] | None = None
        self.image_existed: bool | None = None
        self.is_orchestrator: bool | None = None
        self.permission_mode: str | None = None
        self.agent_id: str | None = None
        self.transcripts: dict[str, list[TranscriptMessage]] = {}

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
        self.messages.append(prompt)
        self.image_paths = image_paths
        self.is_orchestrator = is_orchestrator
        self.permission_mode = permission_mode
        self.agent_id = agent_id
        # Record that the decoded image file exists while the turn runs (the
        # endpoint writes it before this and cleans it up after).
        self.image_existed = bool(image_paths and os.path.isfile(image_paths[0]))
        yield StreamTool(
            tool=ToolCall(server="scufris", tool="host_stats", status="completed")
        )
        yield StreamDone(
            reply=AgentReply(
                text=f"reply: {prompt}",
                status="completed",
                tool_calls=[
                    ToolCall(server="scufris", tool="host_stats", status="completed")
                ],
                usage=TokenUsage(input_tokens=120, output_tokens=8),
            ),
            session_id=session_id or "sess-x",
        )

    def read_status(self, settings: Settings, session_id: str | None) -> None:
        return None

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        if not session_id:
            return []
        return self.transcripts.get(session_id, [])


def _use_fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    """Route every ``get_backend(...)`` in the app to one FakeBackend and return
    it, so a test can assert on the prompts/image it saw."""
    fake = FakeBackend()
    patch_get_backend(monkeypatch, fake)
    return fake


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


def test_stats_endpoint_matches_inspector_output(
    fake_stats: HostStats, fake_collector: Collector, tmp_path: Path
) -> None:
    """`/api/stats` serves the collector's sample VERBATIM, field for field.

    `test_api_stats_returns_snapshot` above spot-checks three values, which a
    move of `HostStats` into another distribution could survive while dropping,
    renaming or reordering everything else. This asserts the WHOLE body against
    the same model serialised directly, so "Stats still serves the same payload"
    is falsifiable across the carve rather than asserted in a task record.
    """
    app = create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))

    resp = TestClient(app).get("/api/stats")

    assert resp.status_code == 200
    assert resp.json() == json.loads(fake_stats.model_dump_json())


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
    # The host overview polls on its own, much slower clock (it shells out).
    assert body["host_overview_seconds"] == 30.0


def test_api_host_overview_returns_the_inspection_snapshot(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The dashboard's host endpoint serves failed units, storage and thermals."""
    app = create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))

    resp = TestClient(app).get("/api/host/overview")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("failed_system_units", "failed_user_units", "storage", "thermal"):
        assert key in body, f"the overview is missing {key}"
    # Both scopes, since scufris itself runs as a USER unit on the real host.
    assert body["failed_system_units"]["scope"] == "system"
    assert body["failed_user_units"]["scope"] == "user"
    # Availability travels to the client, so the UI can show a reason rather
    # than an empty card.
    assert "ok" in body["failed_system_units"]["available"]


def test_host_overview_is_cached(fake_collector: Collector, tmp_path: Path) -> None:
    """N polls from N dashboards must not mean N nixos-rebuild invocations.

    Driven through the real endpoint rather than the cache object, so this covers
    the wiring too: a cache that exists but is not used by the route would pass
    an object-level test and fail here.
    """
    from scufris import app as app_module

    collected = 0
    real_overview = app_module.HostInspector.overview

    def counting_overview(self: app_module.HostInspector) -> HostOverview:
        nonlocal collected
        collected += 1
        return real_overview(self)

    app_module.HostInspector.overview = counting_overview  # type: ignore[method-assign]
    try:
        client = TestClient(
            create_app(
                collector=fake_collector, settings=_settings(tmp_path / "absent")
            )
        )
        for _ in range(5):
            assert client.get("/api/host/overview").status_code == 200
        assert collected == 1, f"the overview was collected {collected} times, not once"
    finally:
        app_module.HostInspector.overview = real_overview  # type: ignore[method-assign]


def test_host_overview_recollects_once_the_ttl_expires() -> None:
    """The cache expires, and does NOT expire early.

    Both halves matter and the second is the one that catches a broken cache: a
    test that only proves "it collected again eventually" also passes against an
    implementation with no cache at all. The clock is injected rather than slept
    on, so the test asserts the boundary instead of a wall-clock guess.
    """
    from scufris_host import HostOverview, HostOverviewCache

    collected = 0

    class CountingInspector:
        def overview(self) -> HostOverview:
            nonlocal collected
            collected += 1
            return HostOverview()

    now = 1000.0
    cache = HostOverviewCache(
        CountingInspector(),  # type: ignore[arg-type]
        ttl_seconds=30.0,
        clock=lambda: now,
    )

    cache.get()
    assert collected == 1

    # Just before expiry: served from cache.
    now = 1029.9
    cache.get()
    assert collected == 1, "the cache expired early"

    # Past expiry: re-collected.
    now = 1030.1
    cache.get()
    assert collected == 2, "the cache never expired"

    # And it caches again from the new timestamp rather than re-collecting.
    cache.get()
    assert collected == 2


def test_host_overview_ttl_has_a_floor_so_zero_does_not_disable_caching() -> None:
    """A misconfigured 0 must not turn every poll of every tab into a subprocess
    fan-out; the endpoint is subprocess-backed and has no business being uncached."""
    from scufris_host import (
        MIN_HOST_OVERVIEW_TTL,
        HostOverview,
        HostOverviewCache,
    )

    collected = 0

    class CountingInspector:
        def overview(self) -> HostOverview:
            nonlocal collected
            collected += 1
            return HostOverview()

    now = 500.0
    cache = HostOverviewCache(
        CountingInspector(),  # type: ignore[arg-type]
        ttl_seconds=0.0,
        clock=lambda: now,
    )
    cache.get()
    now = 500.0 + MIN_HOST_OVERVIEW_TTL / 2
    cache.get()
    assert collected == 1, "a zero TTL disabled caching entirely"


def test_requests_are_logged(
    fake_collector: Collector,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    with caplog.at_level(logging.DEBUG, logger="scufris.api.request_log"):
        TestClient(app).get("/api/config")
    assert any("/api/config -> 200" in record.getMessage() for record in caplog.records)


def test_chat_returns_agent_reply(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use_fake_backend(monkeypatch)
    app = create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    client = TestClient(app)

    resp = client.post("/api/chat", json={"message": "hello agent"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "reply: hello agent"
    assert fake.messages == ["hello agent"]
    # Per-turn metadata rides along with the reply.
    assert body["tool_calls"][0]["tool"] == "host_stats"
    assert body["usage"]["input_tokens"] == 120
    # The landing chat is the orchestrator, so its turn is marked as such (this is
    # what gates the orchestrator-only scufris tools in the codex backend), and it
    # carries the orchestrator's default write posture: auto.
    assert fake.is_orchestrator is True
    assert fake.permission_mode == "auto"


def test_chat_stream_emits_sse_frames(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use_fake_backend(monkeypatch)
    settings = Settings(web_dist=tmp_path / "absent", agent_enabled=True)
    app = create_app(collector=fake_collector, settings=settings)

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
    assert fake.messages == ["hi"]


class _ReasoningBackend:
    """A backend that streams a couple of reasoning deltas then a final answer,
    to prove the turn stream captures the "thinking" into the sidecar (codex only
    streams reasoning; ``name``/agent backend stay codex so the capture gate fires).
    """

    name = "codex"

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
        yield StreamReasoningDelta(delta="let me ")
        yield StreamReasoningDelta(delta="think")
        yield StreamDone(
            reply=AgentReply(text="the answer", status="completed"),
            session_id=session_id or "sess-reason",
        )

    def read_status(self, settings: Settings, session_id: str | None) -> None:
        return None

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        return []


def test_chat_stream_captures_reasoning_to_the_sidecar(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A codex turn's live "thinking" is not on disk, so the turn stream must
    # persist it to the sidecar for reload survival. Drive a turn and confirm the
    # accumulated reasoning landed under the turn's session id.
    backend = _ReasoningBackend()
    patch_get_backend(monkeypatch, backend)
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_backend=Backend.CODEX,
        state_dir=tmp_path / "state",
    )
    app = create_app(collector=fake_collector, settings=settings)

    resp = TestClient(app).post("/api/chat/stream", json={"message": "hi"})
    assert '"kind":"done"' in resp.text

    entries = ReasoningStore(state_database(Path(settings.state_dir))).read(
        "sess-reason"
    )
    assert [e.reasoning for e in entries] == ["let me think"]


# A 1x1 transparent PNG, base64.
_PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_chat_stream_passes_an_attached_image_to_the_agent(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use_fake_backend(monkeypatch)
    settings = Settings(web_dist=tmp_path / "absent", agent_enabled=True)
    app = create_app(collector=fake_collector, settings=settings)

    resp = TestClient(app).post(
        "/api/chat/stream",
        json={
            "message": "what is this?",
            "image": {"data_base64": _PNG_1PX, "mime": "image/png"},
        },
    )
    assert resp.status_code == 200
    assert '"kind":"done"' in resp.text
    # The backend got a real, existing image file path during the turn...
    assert fake.image_paths is not None and len(fake.image_paths) == 1
    assert fake.image_paths[0].endswith(".png")
    assert fake.image_existed is True
    # ...and the temp file is cleaned up afterwards.
    assert not os.path.exists(fake.image_paths[0])


def test_chat_stream_rejects_a_non_image_attachment(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _use_fake_backend(monkeypatch)
    settings = Settings(web_dist=tmp_path / "absent", agent_enabled=True)
    app = create_app(collector=fake_collector, settings=settings)

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
    assert fake.messages == []  # the turn never ran


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
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn runs under the supervisor (decoupled from the request): after the
    stream call, the run is tracked and terminal - proving the endpoint relays a
    supervised run rather than iterating the agent inline (ADR-001)."""
    _use_fake_backend(monkeypatch)
    settings = Settings(web_dist=tmp_path / "absent", agent_enabled=True)
    app = create_app(collector=fake_collector, settings=settings)

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


def test_agent_tools_tagged_by_server(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    )
    body = client.get("/api/agent/tools").json()
    names = {t["name"] for t in body}
    # The console aggregates the orchestrator's two servers: scufris + den.
    assert {"host_stats", "disk_usage", "list_processes"} <= names
    assert {"journal_show", "macros_lookup"} <= names
    assert all(t["description"] for t in body)
    # Each tool reports its SOURCE server (scufris agentic vs den life).
    by_name = {t["name"]: t for t in body}
    assert by_name["host_stats"]["server"] == "scufris"
    assert by_name["journal_show"]["server"] == "den"
    assert set(by_name["list_processes"]["args"]) == {"limit"}


def test_mcp_health_den_warn_when_unconfigured(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/agent/mcp live-probes the orchestrator's two servers: scufris is ok,
    and den is amber (warn) with no den configured, since its journal tools cannot
    actually run - the 'green all good / amber not all good' signal the operator
    asked for."""
    monkeypatch.delenv("SCUFRIS_DEN_PATH", raising=False)
    settings = Settings(
        web_dist=tmp_path / "absent", agent_enabled=True, _env_file=None
    )  # type: ignore[call-arg]
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    servers = {s["id"]: s for s in client.get("/api/agent/mcp").json()}
    assert set(servers) == {"scufris", "den"}
    assert servers["scufris"]["status"] == "ok"
    assert servers["den"]["status"] == "warn"
    assert "not configured" in servers["den"]["detail"]
    # A den tool is listed but marked unavailable (its server cannot run it).
    js = next(t for t in servers["den"]["tools"] if t["name"] == "journal_show")
    assert js["available"] is False and js["enabled"] is True


def test_mcp_health_marks_disabled(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disabled tool reports enabled=false and pushes its server to amber (warn):
    the per-tool bulb goes red and the server dot is no longer green."""
    monkeypatch.delenv("SCUFRIS_DEN_PATH", raising=False)
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        disabled_tools=["disk_usage"],
        _env_file=None,  # type: ignore[call-arg]
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    servers = {s["id"]: s for s in client.get("/api/agent/mcp").json()}
    scufris = servers["scufris"]
    assert scufris["status"] == "warn"
    disk = next(t for t in scufris["tools"] if t["name"] == "disk_usage")
    assert disk["enabled"] is False
    # A non-disabled tool on the same server stays enabled + available.
    host = next(t for t in scufris["tools"] if t["name"] == "host_stats")
    assert host["enabled"] is True and host["available"] is True


def test_agent_tools_endpoint_is_role_scoped(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """GET /api/agents/{id}/tools is audience- AND backend-scoped: the orchestrator
    (on codex) sees its scufris + den surface but not the agent-only callbacks; a
    codex OR claude sub-agent sees ONLY the callback tools `request_input` +
    `report_back` (both backends wire the scufris MCP); a sub-agent whose backend
    does not wire it (mock) sees none; unknown agent 404s. The operator-console
    `/api/agent/tools` is the orchestrator's scufris + den servers."""
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend=Backend.CODEX,  # orchestrator on codex -> has scufris MCP
        enable_mock_backend=True,  # allow creating a mock sub-agent
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    proj = tmp_path / "proj"
    proj.mkdir()
    client.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
    client.post(
        "/api/agents",
        json={"name": "Coder", "project_id": "my-app", "backend": "codex"},
    )
    client.post(
        "/api/agents",
        json={"name": "Clauder", "project_id": "my-app", "backend": "claude"},
    )
    client.post(
        "/api/agents",
        json={"name": "Mocker", "project_id": "my-app", "backend": "mock"},
    )

    # The orchestrator: its scufris + den surface, WITHOUT the agent-only callbacks.
    orch = {
        t["name"] for t in client.get("/api/agents/orchestrator/tools").json()["value"]
    }
    assert {"host_stats", "disk_usage", "list_processes"} <= orch
    assert {"journal_show", "macros_lookup"} <= orch  # den tools
    assert {"request_input", "report_back"}.isdisjoint(orch)

    # A codex sub-agent: ONLY its callback tools, not the orchestrator's surface.
    coder = {t["name"] for t in client.get("/api/agents/coder/tools").json()["value"]}
    assert coder == {"request_input", "report_back"}

    # A claude sub-agent now wires the scufris MCP too -> same role surface.
    clauder = {
        t["name"] for t in client.get("/api/agents/clauder/tools").json()["value"]
    }
    assert clauder == {"request_input", "report_back"}

    # A mock sub-agent: no scufris MCP wiring -> no listing to give at all.
    assert client.get("/api/agents/mocker/tools").json() == {
        "supported": False,
        "value": None,
    }

    # Unknown agent 404s; the operator console is the orchestrator's scufris + den
    # (never the sub-agent callbacks).
    assert client.get("/api/agents/ghost/tools").status_code == 404
    console = {t["name"] for t in client.get("/api/agent/tools").json()}
    assert {"host_stats", "journal_show"} <= console
    assert {"request_input", "report_back"}.isdisjoint(console)


def test_agent_capabilities_endpoint(fake_collector: Collector, tmp_path: Path) -> None:
    """GET /api/agents/{id}/capabilities surfaces the agent's PROJECT-defined
    skills + custom tools, provider-aware; empty for the project-less orchestrator;
    404 for an unknown agent. Read-only."""
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend=Backend.CODEX,
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    proj = tmp_path / "proj"
    (proj / ".claude" / "skills" / "deploy").mkdir(parents=True)
    (proj / ".claude" / "skills" / "deploy" / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: Ship it\n---\n"
    )
    (proj / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"fs": {"command": "npx", "args": ["fs"]}}})
    )
    client.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
    client.post(
        "/api/agents",
        json={"name": "Clauder", "project_id": "my-app", "backend": "claude"},
    )

    caps = client.get("/api/agents/clauder/capabilities")
    assert caps.status_code == 200
    body = caps.json()
    assert [s["name"] for s in body["skills"]] == ["deploy"]
    assert body["skills"][0]["description"] == "Ship it"
    assert [t["name"] for t in body["tools"]] == ["fs"]
    assert body["tools"][0]["kind"] == "stdio"

    # The project-less orchestrator has no project tree -> empty.
    orch = client.get("/api/agents/orchestrator/capabilities").json()
    assert orch == {"skills": [], "tools": []}

    # Unknown agent 404s.
    assert client.get("/api/agents/ghost/capabilities").status_code == 404


def test_agent_health_endpoint_reports_checks(
    fake_collector: Collector, tmp_path: Path
) -> None:
    # A fake codex_bin keeps the probe deterministic (no real codex subprocess).
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_enabled=True,
        agent_backend=Backend.MOCK,
        agent_tools_enabled=True,
        codex_bin=str(tmp_path / "no-such-codex"),
        codex_home=tmp_path / "no-codex",
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    body = client.get("/api/agent/health").json()
    assert body["scufris_version"]
    by_name = {c["name"]: c["status"] for c in body["checks"]}
    assert by_name["agent"] == "ok"
    # The MCP rows follow the ORCHESTRATOR RECORD's backend: mock wires no scufris
    # MCP, so it gets the single "none" row rather than per-server rows.
    assert by_name["mcp tools"] == "warn"
    assert "mcp: scufris" not in by_name
    assert by_name["web assets"] == "error"

    # A backend that DOES wire the scufris MCP gets the per-server rows; the
    # scufris agentic server always advertises tools.
    codex = TestClient(
        create_app(
            collector=fake_collector,
            settings=Settings(
                web_dist=tmp_path / "absent",
                state_dir=tmp_path / "state-codex",
                agent_enabled=True,
                agent_tools_enabled=True,
                codex_bin=str(tmp_path / "no-such-codex"),
                codex_home=tmp_path / "no-codex",
            ),
        )
    )
    codex_checks = {
        c["name"]: c["status"] for c in codex.get("/api/agent/health").json()["checks"]
    }
    assert codex_checks["mcp: scufris"] == "ok"


def test_agent_config_reports_effective_settings(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,  # isolate: the settings-override store must not leak in
        agent_enabled=True,
        agent_backend=Backend.CODEX,
        agent_model="gpt-5.5",
        agent_tools_enabled=True,
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    body = client.get("/api/agent/config").json()
    assert body["backend"] == "codex"
    assert body["model"] == "gpt-5.5"
    assert body["auth_mode"] == "chatgpt"
    assert body["sandbox"] == "read-only"
    assert body["tools_enabled"] is True


def test_agent_config_reports_writable(
    fake_collector: Collector, tmp_path: Path
) -> None:
    writable = Settings(web_dist=tmp_path / "absent", state_dir=tmp_path / "st")
    body = TestClient(create_app(collector=fake_collector, settings=writable)).get(
        "/api/agent/config"
    )
    assert body.json()["writable"] is True
    ro = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path / "st", settings_writable=False
    )
    body = TestClient(create_app(collector=fake_collector, settings=ro)).get(
        "/api/agent/config"
    )
    assert body.json()["writable"] is False


def test_patch_agent_config_persists(fake_collector: Collector, tmp_path: Path) -> None:
    # A non-injected agent so the store's handle path is exercised; mock backend
    # needs no codex. The change must apply and survive into a fresh app.
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend=Backend.MOCK,
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
    assert _override_keys(tmp_path) == {"agent_model", "agent_tools_enabled"}

    fresh = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend=Backend.MOCK
    )
    body2 = (
        TestClient(create_app(collector=fake_collector, settings=fresh))
        .get("/api/agent/config")
        .json()
    )
    assert body2["model"] == "gpt-5.6"
    assert body2["tools_enabled"] is False


def test_patch_agent_config_backend_change_clears_orchestrator_session(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """Switching the orchestrator's backend at runtime drops its active session,
    so a stale cross-backend session id is never resumed under the new backend
    (the settings-store on_change wiring that replaced AgentHandle's session
    carry - now it CLEARS on a backend switch instead of carrying)."""
    settings = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend=Backend.MOCK
    )
    app = create_app(collector=fake_collector, settings=settings)
    app.state.agents.set_orchestrator_session("mock-session-live")
    client = TestClient(app)

    resp = client.patch("/api/agent/config", json={"agent_backend": "codex"})
    assert resp.status_code == 200
    assert resp.json()["backend"] == "codex"
    # The stale mock session is gone after the switch to codex.
    assert app.state.agents.orchestrator_session_id() is None


def test_patch_agent_config_rejects_legacy_backend_id(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The API input model is STRICT: a legacy codex-mode id (`app_server`) is
    rejected with 422 on a new write, even though a persisted/env `app_server`
    still coerces to `codex` on load. New writes must use the canonical vocab."""
    settings = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend=Backend.CODEX
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    resp = client.patch("/api/agent/config", json={"agent_backend": "app_server"})
    assert resp.status_code == 422


def test_patch_agent_config_forbidden_when_readonly(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        settings_writable=False,
        agent_backend=Backend.MOCK,
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    resp = client.patch("/api/agent/config", json={"agent_model": "gpt-5.6"})
    assert resp.status_code == 403
    assert not (tmp_path / "settings.json").exists()


def test_patch_agent_config_rejects_non_whitelisted(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend=Backend.MOCK
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
        disabled_tools=["disk_usage"],
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    tools = {t["name"]: t["enabled"] for t in client.get("/api/agent/tools").json()}
    assert tools["disk_usage"] is False
    assert tools["host_stats"] is True


def test_patch_disabled_tools_persists(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend=Backend.MOCK
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    resp = client.patch("/api/agent/config", json={"disabled_tools": ["disk_usage"]})
    assert resp.status_code == 200
    # A fresh app over the same state dir marks it disabled in the tools list.
    fresh = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend=Backend.MOCK
    )
    tools = {
        t["name"]: t["enabled"]
        for t in TestClient(create_app(collector=fake_collector, settings=fresh))
        .get("/api/agent/tools")
        .json()
    }
    assert tools["disk_usage"] is False


def test_orchestrator_permission_mode_defaults_auto_and_edit_persists(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The orchestrator's write posture defaults to auto, and changing it via the
    unified agent PATCH lands in the settings store, surviving an app restart."""
    settings = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend=Backend.MOCK
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    assert client.get("/api/agents/orchestrator").json()["permission_mode"] == "auto"

    resp = client.patch("/api/agents/orchestrator", json={"permission_mode": "manual"})
    assert resp.status_code == 200
    assert resp.json()["permission_mode"] == "manual"

    # A fresh app over the same state dir still reads the edited mode.
    fresh = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, agent_backend=Backend.MOCK
    )
    fresh_client = TestClient(create_app(collector=fake_collector, settings=fresh))
    assert (
        fresh_client.get("/api/agents/orchestrator").json()["permission_mode"]
        == "manual"
    )


def test_tools_endpoint_exposes_parameters(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The tools list carries a typed param schema for the 'try it' runner."""
    client = TestClient(
        create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    )
    body = client.get("/api/agent/tools").json()
    by_name = {t["name"]: t for t in body}
    # list_processes.limit is an integer with a default (not required).
    lp = {p["name"]: p for p in by_name["list_processes"]["parameters"]}
    assert lp["limit"]["type"] == "integer"
    assert lp["limit"]["required"] is False
    # agent_status.agent_id is a required string.
    ast = {p["name"]: p for p in by_name["agent_status"]["parameters"]}
    assert ast["agent_id"]["type"] == "string"
    assert ast["agent_id"]["required"] is True


def test_tool_parameters_handles_malformed_schema() -> None:
    """`tool_parameters` is best-effort: a malformed schema yields [], never raises."""
    from scufris.agent_diagnostics import tool_parameters as _tool_parameters

    assert _tool_parameters(None) == []  # not a dict
    assert _tool_parameters({"type": "object"}) == []  # no properties
    assert _tool_parameters({"properties": "nope"}) == []  # properties not a dict
    # A non-dict property spec falls back to a string param, still no raise.
    params = _tool_parameters({"properties": {"x": "not-a-dict"}})
    assert [(p.name, p.type, p.required) for p in params] == [("x", "string", False)]


def test_run_tool_host_stats_returns_result(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """POST .../run executes one MCP tool in-process and returns its output."""
    client = TestClient(
        create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    )
    resp = client.post("/api/agent/tools/host_stats/run", json={"args": {}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    # host_stats returns a JSON blob with the hostname - a real, populated result.
    assert "hostname" in body["text"]
    # The structured block is populated too - the contract the frontend runner reads.
    assert body["structured"].get("hostname")


def test_run_tool_rejects_disabled_unknown_and_badargs(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The runner refuses a disabled tool (403), unknown tool (404), and bad
    args (422) - never an uncontrolled 500."""
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        disabled_tools=["host_stats"],
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))

    disabled = client.post("/api/agent/tools/host_stats/run", json={"args": {}})
    assert disabled.status_code == 403

    unknown = client.post("/api/agent/tools/does_not_exist/run", json={"args": {}})
    assert unknown.status_code == 404

    # list_processes.limit expects an integer; a string is a validation error.
    bad = client.post(
        "/api/agent/tools/list_processes/run",
        json={"args": {"limit": "notanint"}},
    )
    assert bad.status_code == 422

    # A host-inspection tool through the SECOND runner: the operator console runs
    # it IN-PROCESS, where an agent turn runs it in an MCP subprocess. A tool can
    # pass one and fail the other (see the ledger,
    # tool-reachable-by-two-runners-needs-a-test-per-runner), so the console path
    # gets its own assertion rather than trusting the direct-call test.
    host = client.post("/api/agent/tools/host_failed_units/run", json={"args": {}})
    assert host.status_code == 200
    text = json.dumps(host.json())
    assert "system units" in text


def test_ensure_den_path_bridges_settings_into_env() -> None:
    """ensure_den_path exports settings.den_path to SCUFRIS_DEN_PATH so an in-process
    console run resolves the den; setdefault means an explicit env wins and an unset
    den is a no-op. It mutates os.environ directly, so snapshot/restore the key
    (setdefault leaks past monkeypatch - see the ledger)."""
    saved = os.environ.pop("SCUFRIS_DEN_PATH", None)
    try:
        # absent -> bridged from settings
        ensure_den_path(Settings(den_path=Path("/home/op/the-den"), _env_file=None))  # type: ignore[call-arg]
        assert os.environ["SCUFRIS_DEN_PATH"] == "/home/op/the-den"
        # explicit env wins (setdefault no-op) - the deployed service sets it directly
        os.environ["SCUFRIS_DEN_PATH"] = "/explicit/den"
        ensure_den_path(Settings(den_path=Path("/home/op/the-den"), _env_file=None))  # type: ignore[call-arg]
        assert os.environ["SCUFRIS_DEN_PATH"] == "/explicit/den"
        # unset den -> no bridge (tools stay correctly inert)
        del os.environ["SCUFRIS_DEN_PATH"]
        ensure_den_path(Settings(den_path=None, _env_file=None))  # type: ignore[call-arg]
        assert "SCUFRIS_DEN_PATH" not in os.environ
    finally:
        os.environ.pop("SCUFRIS_DEN_PATH", None)
        if saved is not None:
            os.environ["SCUFRIS_DEN_PATH"] = saved


def test_journal_tool_from_console_bridges_den(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The operator console runs journal_show IN-PROCESS; with a den configured it
    must NOT report "not configured" - the fix bridges the den env before running.
    (Whether the real `today` CLI is on PATH is orthogonal: without it the tool
    returns "today not found on PATH", still proving the den gate PASSED. Without the
    bridge it returns the den-not-configured error - the bug.) The endpoint
    setdefaults the env key, so snapshot/restore it."""
    den = tmp_path / "the-den"
    den.mkdir()
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        den_path=den,
        _env_file=None,  # type: ignore[call-arg]
    )
    saved = os.environ.pop("SCUFRIS_DEN_PATH", None)
    try:
        client = TestClient(create_app(collector=fake_collector, settings=settings))
        resp = client.post("/api/agent/tools/journal_show/run", json={"args": {}})
        assert resp.status_code == 200
        text = resp.json()["text"]
        assert "not configured" not in text, text
    finally:
        os.environ.pop("SCUFRIS_DEN_PATH", None)
        if saved is not None:
            os.environ["SCUFRIS_DEN_PATH"] = saved


def _override_keys(state_dir: Path) -> set[str]:
    """Which settings the store has persisted an override for.

    The overrides are `settings_override` rows now, so "did that stick" is a
    query on the same database the app wrote to rather than a file check.
    """
    from sqlalchemy import select

    from scufris.db.models import SettingsOverrideRow

    with state_database(state_dir).transaction() as conn:
        return set(conn.execute(select(SettingsOverrideRow.key)).scalars())


def _mock_settings(tmp_path: Path) -> Settings:
    return Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend=Backend.MOCK,
        enable_mock_backend=True,  # allow creating mock-backed agent records
    )


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


def _disco_settings(tmp_path: Path, base: Path, *, writable: bool = True) -> Settings:
    return Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path / "st",
        project_base_dirs=[base],
        settings_writable=writable,
    )


def test_projects_discovered_unions_registered(
    fake_collector: Collector, tmp_path: Path
) -> None:
    base = tmp_path / "personal"
    (base / "api").mkdir(parents=True)
    (base / "api" / "pyproject.toml").write_text("")
    (base / "web").mkdir()  # discovered, not registered
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    client = TestClient(
        create_app(collector=fake_collector, settings=_disco_settings(tmp_path, base))
    )
    # Register one discovered dir and one dir OUTSIDE the base.
    api = client.post("/api/projects", json={"name": "Api", "cwd": str(base / "api")})
    assert api.status_code == 200
    client.post("/api/projects", json={"name": "Ext", "cwd": str(outside)})

    payload = client.get("/api/projects/discovered").json()
    disco = {d["name"]: d for d in payload["projects"]}
    # The discovered-but-unregistered dir shows up, unmarked.
    assert disco["web"]["registered"] is False and disco["web"]["project_id"] is None
    # The registered discovered dir is marked with its language + id.
    assert disco["api"]["registered"] is True
    assert disco["api"]["project_id"] == api.json()["id"]
    assert disco["api"]["language"] == "python"
    # A registered project OUTSIDE the base dirs is still surfaced (registered).
    assert disco["Ext"]["registered"] is True
    assert disco["Ext"]["path"] == str(outside.resolve())
    # The base dirs ride along for the create form's picker.
    assert payload["base_dirs"] == [str(base)]


def test_project_new_mkdirs_and_registers(
    fake_collector: Collector, tmp_path: Path
) -> None:
    base = tmp_path / "personal"
    base.mkdir()
    client = TestClient(
        create_app(collector=fake_collector, settings=_disco_settings(tmp_path, base))
    )
    resp = client.post("/api/projects/new", json={"name": "fresh", "base": str(base)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cwd"] == str((base / "fresh").resolve())
    assert (base / "fresh").is_dir()  # the directory was created
    # And it is now registered (discovered lists it as registered).
    disco = {
        d["name"]: d for d in client.get("/api/projects/discovered").json()["projects"]
    }
    assert disco["fresh"]["registered"] is True


def test_project_new_rejects_base_outside_allowed_and_unsafe_name(
    fake_collector: Collector, tmp_path: Path
) -> None:
    base = tmp_path / "personal"
    base.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    client = TestClient(
        create_app(collector=fake_collector, settings=_disco_settings(tmp_path, base))
    )
    # A base not in project_base_dirs is refused (no mkdir outside the allowed set).
    outside = client.post(
        "/api/projects/new", json={"name": "x", "base": str(elsewhere)}
    )
    assert outside.status_code == 422
    assert not (elsewhere / "x").exists()
    # A traversing name is refused.
    bad = client.post(
        "/api/projects/new", json={"name": "../escape", "base": str(base)}
    )
    assert bad.status_code == 422


def test_project_new_forbidden_when_readonly(
    fake_collector: Collector, tmp_path: Path
) -> None:
    base = tmp_path / "personal"
    base.mkdir()
    client = TestClient(
        create_app(
            collector=fake_collector,
            settings=_disco_settings(tmp_path, base, writable=False),
        )
    )
    resp = client.post("/api/projects/new", json={"name": "fresh", "base": str(base)})
    assert resp.status_code == 403
    assert not (base / "fresh").exists()  # no mkdir side-effect on a refused write


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
        agent_backend=Backend.MOCK,
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
    app = create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    # Seed an active orchestrator session; reset must clear it.
    app.state.agents.set_orchestrator_session("sess-live")
    client = TestClient(app)

    resp = client.post("/api/chat/reset")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert app.state.agents.orchestrator_session_id() is None


def _agent_settings(web_dist: Path, codex_home: Path) -> Settings:
    return Settings(web_dist=web_dist, agent_enabled=True, codex_home=codex_home)


def _claude_agent_settings(web_dist: Path, claude_home: Path) -> Settings:
    """A claude-backed orchestrator, to prove the session endpoints route through
    the orchestrator's backend rather than the codex home."""
    return Settings(
        web_dist=web_dist,
        agent_enabled=True,
        agent_backend=Backend.CLAUDE,
        claude_home=claude_home,
        state_dir=claude_home.parent / "state",
    )


def _write_claude_session(claude_home: Path, session_id: str) -> Path:
    """A minimal claude transcript file (mirrors the backend test fixture)."""
    proj = claude_home / "projects" / "-proj"
    proj.mkdir(parents=True, exist_ok=True)
    path = proj / f"{session_id}.jsonl"
    lines = [
        {"type": "user", "message": {"role": "user", "content": "hello claude"}},
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "hi from claude"}],
                "usage": {"input_tokens": 50, "output_tokens": 7},
            },
        },
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


def test_orchestrator_transcript_uses_backend(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A claude-backed orchestrator re-renders a session transcript via the claude
    backend, not the codex home - so switching into a session works off codex."""
    home = tmp_path / "claude"
    _write_claude_session(home, "cl-sess")
    app = create_app(
        collector=fake_collector,
        settings=_claude_agent_settings(tmp_path / "absent", home),
    )
    body = TestClient(app).get("/api/agent/session/cl-sess").json()
    assert [m["text"] for m in body["messages"]] == ["hello claude", "hi from claude"]


def test_orchestrator_context_uses_backend(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A claude-backed orchestrator's context readout comes from the claude
    backend (mapped from status; window 0)."""
    home = tmp_path / "claude"
    _write_claude_session(home, "cl-ctx")
    app = create_app(
        collector=fake_collector,
        settings=_claude_agent_settings(tmp_path / "absent", home),
    )
    app.state.agents.set_orchestrator_session("cl-ctx")
    body = TestClient(app).get("/api/agent/context").json()
    assert body["session_id"] == "cl-ctx"
    assert body["turn_count"] == 1
    assert body["input_tokens"] == 50
    assert body["context_window"] == 0  # claude exposes no window


def test_orchestrator_delete_uses_backend(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """Deleting a session on a claude-backed orchestrator unlinks the claude file
    (provider delete) AND forgets it from the switcher, off codex entirely."""
    home = tmp_path / "claude"
    path = _write_claude_session(home, "cl-del")
    app = create_app(
        collector=fake_collector,
        settings=_claude_agent_settings(tmp_path / "absent", home),
    )
    app.state.agents.set_orchestrator_session("cl-del")
    resp = TestClient(app).delete("/api/agent/session/cl-del")
    assert resp.json()["deleted"] is True
    assert not path.exists()  # the claude transcript file is gone
    assert app.state.agents.orchestrator_session_id() is None


def test_orchestrator_fork_uses_backend_transcript(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """Fork seeds from the orchestrator backend's transcript. A mock-backed
    orchestrator returns an empty transcript, so the fork seed is just the edited
    text - proving fork no longer reads the codex home."""
    app = create_app(
        collector=fake_collector,
        settings=Settings(
            web_dist=tmp_path / "absent",
            agent_enabled=True,
            agent_backend=Backend.MOCK,
            enable_mock_backend=True,
            state_dir=tmp_path / "state",
        ),
    )
    resp = TestClient(app).post(
        "/api/agent/session/fork",
        json={"source_id": "whatever", "message_index": 0, "text": "forked prompt"},
    )
    assert resp.status_code == 200
    # The mock backend echoes the seed as its reply; with an empty transcript the
    # seed is exactly the edited text.
    assert "forked prompt" in resp.json()["reply"]["text"]


def test_sessions_lists_and_reports_current(
    fake_collector: Collector, tmp_path: Path
) -> None:
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-1", cwd=os.getcwd())
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
    )
    app.state.agents.set_orchestrator_session("sess-1")
    body = TestClient(app).get("/api/agent/sessions").json()
    assert body["current"] == "sess-1"
    assert [s["id"] for s in body["sessions"]] == ["sess-1"]
    assert body["sessions"][0]["title"] == "list my tasks"


def test_sessions_lists_a_just_started_session_with_no_user_message(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A just-started codex thread (rollout has only ``session_meta``, no user
    message flushed yet) must still appear in the switcher as "(untitled)" - so a
    mid-turn refresh sees the in-flight session rather than dropping it. Guards the
    turn-start recording: `session_info` returns a row when a status snapshot is
    readable even with an empty transcript."""
    home = tmp_path / "codex"
    day = home / "sessions" / "2026" / "07" / "24"
    day.mkdir(parents=True, exist_ok=True)
    # Only the session_meta line codex writes at thread/start - no user_message.
    (day / "rollout-2026-07-24T10-00-00-sess-fresh.jsonl").write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "session_id": "sess-fresh",
                    "id": "sess-fresh",
                    "timestamp": "2026-07-24T10:00:00.000Z",
                    "cwd": os.getcwd(),
                    "originator": "scufris",
                    "git": {"branch": "main"},
                },
            }
        )
        + "\n"
    )
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
    )
    app.state.agents.set_orchestrator_session("sess-fresh")
    body = TestClient(app).get("/api/agent/sessions").json()
    assert body["current"] == "sess-fresh"
    assert [s["id"] for s in body["sessions"]] == ["sess-fresh"]
    assert body["sessions"][0]["title"] == "(untitled)"


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
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", tmp_path / "codex"),
    )
    client = TestClient(app)

    switched = client.post(
        "/api/agent/session", json={"action": "switch", "session_id": "sess-9"}
    )
    assert switched.status_code == 200
    assert switched.json()["current"] == "sess-9"
    assert app.state.agents.orchestrator_session_id() == "sess-9"

    fresh = client.post("/api/agent/session", json={"action": "new"})
    assert fresh.json()["current"] is None


def test_session_switch_requires_id(fake_collector: Collector, tmp_path: Path) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", tmp_path / "codex"),
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
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
    )
    app.state.agents.set_orchestrator_session("sess-ctx")
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
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
    )
    app.state.agents.set_orchestrator_session("sess-del")
    client = TestClient(app)

    resp = client.delete("/api/agent/session/sess-del")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] is True
    assert body["current"] is None  # was the active session -> reset
    # It is gone from the list.
    listed = client.get("/api/agent/sessions").json()["sessions"]
    assert listed == []


def test_orchestrator_switcher_excludes_subagent_sessions(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The leak repro (part 1, spike 20260724-111839): a codex sub-agent's chat
    sits on disk in the same home/cwd as the orchestrator (same originator, same
    cwd), but the switcher is driven by the ownership registry - so only the
    orchestrator's OWN session appears, never the sub-agent's."""
    home = tmp_path / "codex"
    _write_session_rollout(home, "orch-sess", cwd=os.getcwd())
    _write_session_rollout(home, "sub-sess", cwd=os.getcwd())  # a sub-agent's chat
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
    )
    app.state.agents.set_orchestrator_session("orch-sess")
    listed = TestClient(app).get("/api/agent/sessions").json()["sessions"]
    assert [s["id"] for s in listed] == ["orch-sess"]  # sub-sess must NOT leak in


def test_orchestrator_switcher_lists_registry_history(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The switcher shows every session the registry attributes to the
    orchestrator, not just the current one - multi-session driven by the index."""
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-old", cwd=os.getcwd())
    _write_session_rollout(home, "sess-new", cwd=os.getcwd())
    # Distinct mtimes so the "newest first" ordering is actually exercised
    # (updated_at is the rollout's mtime): sess-new is more recent than sess-old.
    day = home / "sessions" / "2026" / "07" / "19"
    os.utime(day / "rollout-2026-07-19T14-39-30-sess-old.jsonl", (1_000, 1_000))
    os.utime(day / "rollout-2026-07-19T14-39-30-sess-new.jsonl", (2_000, 2_000))
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
    )
    agents = app.state.agents
    agents.set_orchestrator_session("sess-old")
    agents.set_orchestrator_session(None)  # new chat - keeps history
    agents.set_orchestrator_session("sess-new")
    listed = TestClient(app).get("/api/agent/sessions").json()
    # Both listed, and newest (by mtime) first.
    assert [s["id"] for s in listed["sessions"]] == ["sess-new", "sess-old"]
    assert listed["current"] == "sess-new"


def test_new_chat_preserves_session_history(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """Starting a new chat clears only `current`; prior sessions stay listed."""
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-1", cwd=os.getcwd())
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
    )
    agents = app.state.agents
    agents.set_orchestrator_session("sess-1")
    agents.set_orchestrator_session(None)  # new chat
    assert agents.orchestrator_session_id() is None
    listed = TestClient(app).get("/api/agent/sessions").json()["sessions"]
    assert [s["id"] for s in listed] == ["sess-1"]  # history preserved


def test_delete_session_keeps_current_when_other(
    fake_collector: Collector, tmp_path: Path
) -> None:
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-a", cwd=os.getcwd())
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
    )
    app.state.agents.set_orchestrator_session("sess-current")
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
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    fake = _use_fake_backend(monkeypatch)
    fake.transcripts["sess-src"] = [
        TranscriptMessage(role="user", text="first question"),
        TranscriptMessage(role="assistant", text="first answer"),
        TranscriptMessage(role="user", text="second question"),
    ]
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
    )
    # Fork at the second user message (index 2), editing its text.
    resp = TestClient(app).post(
        "/api/agent/session/fork",
        json={"source_id": "sess-src", "message_index": 2, "text": "edited second"},
    )
    assert resp.status_code == 200
    # The seed prompt (what the backend was asked) carries the prior turns + edit.
    seed = fake.messages[-1]
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
    )
    body = TestClient(app).get("/api/agent/usage").json()
    assert body["supported"] is True
    assert body["value"]["plan_type"] == "plus"
    assert body["value"]["primary"]["window_minutes"] == 10080
    assert body["value"]["primary"]["used_percent"] == 42.0


def test_disabled_agent_is_supported_not_unsupported(
    fake_collector: Collector, tmp_path: Path
) -> None:
    # DECISION-4: disabling the agent does not remove a capability. The codex
    # backend still HAS the usage and memory readers and still DELEGATES to
    # them, so a POPULATED home reports its real reading even while disabled.
    # Only `enabled` on the account carries the disabled state. The home is
    # seeded on purpose: an empty one reads the same as a short-circuit that
    # skips the readers, so it would pin nothing (see falsify.sh, R2.1).
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-d", cwd=os.getcwd(), used_percent=42.0)
    app = create_app(
        collector=fake_collector,
        settings=Settings(
            web_dist=tmp_path / "absent",
            state_dir=tmp_path / "state",
            codex_home=home,
            agent_enabled=False,
        ),
    )
    client = TestClient(app)
    usage = client.get("/api/agent/usage").json()
    assert usage["supported"] is True
    assert usage["value"]["primary"]["used_percent"] == 42.0

    memory = client.get("/api/agent/memory").json()
    assert memory["supported"] is True
    assert memory["value"]["session_count"] == 1

    account = client.get("/api/agent/account").json()
    assert account["enabled"] is False
    assert account["quota"]["supported"] is True
    assert account["quota"]["value"]["primary"]["used_percent"] == 42.0


def test_memory_endpoint_reports_footprint(
    fake_collector: Collector, tmp_path: Path
) -> None:
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-a", cwd=os.getcwd())
    _write_session_rollout(home, "sess-b", cwd=os.getcwd())
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
    )
    body = TestClient(app).get("/api/agent/memory").json()
    assert body["supported"] is True
    assert body["value"]["session_count"] == 2
    assert body["value"]["total_bytes"] > 0
    assert body["value"]["oldest"] is not None and body["value"]["newest"] is not None


def test_memory_endpoint_empty_ok(fake_collector: Collector, tmp_path: Path) -> None:
    # Missing sessions dir -> a supported reading of zeros, not an error and not
    # an unsupported capability: the codex backend does have the reader.
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", tmp_path / "no-codex"),
    )
    body = TestClient(app).get("/api/agent/memory").json()
    assert body == {
        "supported": True,
        "value": {
            "session_count": 0,
            "total_bytes": 0,
            "oldest": None,
            "newest": None,
        },
    }


def test_account_endpoint_shape(fake_collector: Collector, tmp_path: Path) -> None:
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-acc", cwd=os.getcwd(), used_percent=17.0)
    app = create_app(
        collector=fake_collector,
        settings=_agent_settings(tmp_path / "absent", home),
    )
    body = TestClient(app).get("/api/agent/account").json()
    assert body["auth_mode"] == "chatgpt"
    assert body["model"]  # non-empty
    assert body["enabled"] is True
    assert body["quota"]["value"]["primary"]["used_percent"] == 17.0


def test_account_quota_empty_reading_when_disabled(
    fake_collector: Collector, tmp_path: Path
) -> None:
    app = create_app(
        collector=fake_collector,
        settings=Settings(
            web_dist=tmp_path / "absent",
            state_dir=tmp_path / "state",
            codex_home=tmp_path / "no-codex",
            agent_enabled=False,
        ),
    )
    body = TestClient(app).get("/api/agent/account").json()
    assert body["enabled"] is False
    # Disabled is not unsupported: the codex backend still has a usage reader,
    # so the envelope stays supported with no value.
    assert body["quota"] == {"supported": True, "value": None}


def test_per_agent_panels_dispatch_by_backend(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """/api/agents/{id}/usage|memory|account resolve per agent and dispatch by
    backend: real codex-account data for a codex agent (and the orchestrator),
    an unsupported capability for a non-codex (mock) agent; 404 for an unknown id.
    The cross-backend contract itself is pinned in test_agent_diagnostics."""
    home = tmp_path / "codex"
    _write_session_rollout(home, "sess-p", cwd=os.getcwd(), used_percent=37.0)
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path / "state",
        codex_home=home,
        agent_enabled=True,
        enable_mock_backend=True,  # to create a mock (non-codex) agent
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    client.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
    cx = client.post(
        "/api/agents",
        # An explicit, non-default model so `account.model` proves it returns the
        # AGENT's effective model, not the global `settings.agent_model`.
        json={
            "name": "Cx",
            "project_id": "my-app",
            "backend": "codex",
            "model": "gpt-5-codex-custom",
        },
    ).json()["id"]
    mk = client.post(
        "/api/agents",
        json={"name": "Mk", "project_id": "my-app", "backend": "mock"},
    ).json()["id"]

    # A codex agent sees the real codex-account data.
    usage = client.get(f"/api/agents/{cx}/usage").json()
    assert usage["value"]["primary"]["used_percent"] == 37.0
    assert client.get(f"/api/agents/{cx}/memory").json()["value"]["session_count"] >= 1
    acct = client.get(f"/api/agents/{cx}/account").json()
    assert acct["model"] == "gpt-5-codex-custom"  # the agent's model, not global
    assert acct["quota"]["value"] is not None
    # The orchestrator (codex default) resolves the same panels.
    assert client.get("/api/agents/orchestrator/usage").json()["value"] is not None

    # A mock agent has no codex account -> unsupported (not an error, not a zero).
    unsupported = {"supported": False, "value": None}
    assert client.get(f"/api/agents/{mk}/usage").json() == unsupported
    assert client.get(f"/api/agents/{mk}/memory").json() == unsupported
    assert client.get(f"/api/agents/{mk}/account").json()["quota"] == unsupported

    # Unknown agent id -> 404 on every panel.
    for panel in ("usage", "memory", "account"):
        assert client.get(f"/api/agents/ghost/{panel}").status_code == 404


def _orchestrator_client(
    fake_collector: Collector, tmp_path: Path, backend: Backend
) -> TestClient:
    """A client whose ORCHESTRATOR runs on ``backend``, with a populated codex
    home: any codex-account data reaching the wire for a non-codex orchestrator is
    a leak, not a coincidence."""
    home = tmp_path / "codex"
    if not home.exists():
        _write_session_rollout(home, "sess-leak", cwd=os.getcwd(), used_percent=88.0)
    return TestClient(
        create_app(
            collector=fake_collector,
            settings=Settings(
                web_dist=tmp_path / "absent",
                state_dir=tmp_path / f"state-{backend}",
                codex_home=home,
                agent_backend=backend,
                agent_enabled=True,
                enable_mock_backend=True,
            ),
        )
    )


@pytest.mark.parametrize(
    "backend", [Backend.CODEX, Backend.CLAUDE, Backend.OPENCODE, Backend.MOCK]
)
def test_orchestrator_surfaces_are_backend_consistent(
    fake_collector: Collector, tmp_path: Path, backend: Backend
) -> None:
    """The legacy singular `/api/agent/*` family and the scoped
    `/api/agents/orchestrator/*` family describe the SAME agent, so for every
    backend they must agree: same effective model, auth mode, capability
    envelopes and probed backend. Red before delegation: a claude orchestrator
    reports the codex model, a codex quota and a codex footprint."""
    client = _orchestrator_client(fake_collector, tmp_path, backend)

    assert client.get("/api/agent/usage").json() == (
        client.get("/api/agents/orchestrator/usage").json()
    )
    assert client.get("/api/agent/memory").json() == (
        client.get("/api/agents/orchestrator/memory").json()
    )
    account = client.get("/api/agents/orchestrator/account").json()
    assert client.get("/api/agent/account").json() == account

    # `/api/agent/info` and `/api/agent/config` describe the same account.
    info = client.get("/api/agent/info").json()
    config = client.get("/api/agent/config").json()
    assert info["model"] == account["model"] == config["model"]
    assert info["auth_mode"] == account["auth_mode"] == config["auth_mode"]
    assert info["enabled"] == account["enabled"] == config["enabled"]

    # Health probes the record's backend and scopes the same MCP rows.
    legacy_health = client.get("/api/agent/health").json()
    scoped_health = client.get("/api/agents/orchestrator/health").json()
    assert legacy_health["backend"] == scoped_health["backend"]
    assert [c["name"] for c in legacy_health["checks"]] == [
        c["name"] for c in scoped_health["checks"]
    ]

    # Deliberate divergence: the console's own in-process tool surface keeps
    # listing tools even where the agent-scoped route reports no listing at all.
    assert client.get("/api/agent/tools").json()
    assert client.get("/api/agent/mcp").json()
    if backend in (Backend.OPENCODE, Backend.MOCK):
        assert client.get("/api/agents/orchestrator/tools").json() == {
            "supported": False,
            "value": None,
        }


def test_legacy_agent_routes_delegate_to_scoped_diagnostics(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A non-codex orchestrator must not serve the codex rollouts sitting on disk:
    usage and memory report `supported: false` and the account quota with them."""
    client = _orchestrator_client(fake_collector, tmp_path, Backend.CLAUDE)
    unsupported = {"supported": False, "value": None}

    assert client.get("/api/agent/usage").json() == unsupported
    assert client.get("/api/agent/memory").json() == unsupported
    assert client.get("/api/agent/account").json()["quota"] == unsupported


def test_per_agent_account_auth_mode_dispatches_by_backend(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The account panel's auth_mode is the agent's OWN backend's mode: chatgpt for
    a codex agent, claude_ai for a claude agent, None for a mock agent - not the
    flat global value. A claude agent must not report the codex auth."""
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path / "state",
        agent_enabled=True,
        enable_mock_backend=True,
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    client.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
    ids = {
        backend: client.post(
            "/api/agents",
            json={"name": backend, "project_id": "my-app", "backend": backend},
        ).json()["id"]
        for backend in ("codex", "claude", "opencode", "mock")
    }

    def auth(agent_id: str) -> object:
        return client.get(f"/api/agents/{agent_id}/account").json()["auth_mode"]

    assert auth(ids["codex"]) == "chatgpt"
    assert auth(ids["claude"]) == "claude_ai"  # the fix: claude, not codex
    assert auth(ids["opencode"]) == "local"  # self-hosted, no login
    assert auth(ids["mock"]) is None  # no login modeled


def test_per_agent_account_auth_mode_respects_claude_api_key(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A claude agent reports api_key when the claude auth mode is configured so."""
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path / "state",
        agent_enabled=True,
        agent_claude_auth_mode=AuthMode.API_KEY,
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    client.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
    cl = client.post(
        "/api/agents",
        json={"name": "Cl", "project_id": "my-app", "backend": "claude"},
    ).json()["id"]
    assert client.get(f"/api/agents/{cl}/account").json()["auth_mode"] == "api_key"


def test_per_agent_health_probes_the_agents_backend(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """GET /api/agents/{id}/health probes THIS agent's backend: a claude agent
    reports claude checks (not codex) even though the server default is codex; a
    codex agent reports codex; the orchestrator resolves; an unknown id 404s.
    Fake bins keep the probes deterministic (no real subprocess)."""
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path / "state",
        agent_enabled=True,
        agent_backend=Backend.CODEX,  # the server/orchestrator default is codex
        agent_tools_enabled=True,
        codex_bin=str(tmp_path / "no-such-codex"),
        claude_bin=str(tmp_path / "no-such-claude"),
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    client.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
    cx = client.post(
        "/api/agents",
        json={"name": "Cx", "project_id": "my-app", "backend": "codex"},
    ).json()["id"]
    cl = client.post(
        "/api/agents",
        json={"name": "Cl", "project_id": "my-app", "backend": "claude"},
    ).json()["id"]

    # The claude agent's health probes claude, with NO codex checks - the bug fix.
    cl_body = client.get(f"/api/agents/{cl}/health").json()
    assert cl_body["backend"] == "claude"
    cl_checks = {c["name"] for c in cl_body["checks"]}
    assert "claude cli" in cl_checks
    assert "codex cli" not in cl_checks and "codex auth" not in cl_checks

    # The codex agent's health probes codex.
    cx_body = client.get(f"/api/agents/{cx}/health").json()
    assert cx_body["backend"] == "codex"
    cx_checks = {c["name"] for c in cx_body["checks"]}
    assert "codex cli" in cx_checks
    assert "claude cli" not in cx_checks

    # The orchestrator resolves (server default backend), and the field is neutral.
    orch = client.get("/api/agents/orchestrator/health").json()
    assert orch["backend"] == "codex"
    assert "codex_version" not in orch  # the codex-specific field name is gone

    assert client.get("/api/agents/ghost/health").status_code == 404


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


def _non_reserved(agents: list[dict]) -> list[dict]:
    """The project-bound agents: everything but the two synthetic reserved records
    (the hidden orchestrator and the listed host agent)."""
    return [a for a in agents if a["id"] not in ("orchestrator", "host")]


def test_agents_crud_endpoints(fake_collector: Collector, tmp_path: Path) -> None:
    client = _client_with_project(fake_collector, tmp_path)
    # The reserved orchestrator is always present; no project agents yet.
    assert _non_reserved(client.get("/api/agents").json()) == []

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
    assert _non_reserved(client.get("/api/agents").json()) == []
    assert client.delete("/api/agents/ghost").status_code == 404


def test_orchestrator_reserved_via_api(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The reserved orchestrator is a HIDDEN default: excluded from the /agents
    list, resolvable at /api/agents/orchestrator, projectless, and undeletable
    (403). Its edits route to the settings store (see the dedicated edit test)."""
    client = _client_with_project(fake_collector, tmp_path)  # agent_backend=mock
    ids = [a["id"] for a in client.get("/api/agents").json()]
    assert "orchestrator" not in ids  # hidden from the list
    orch = client.get("/api/agents/orchestrator").json()  # but resolvable
    assert orch["project_id"] == ""  # no project
    assert orch["backend"] == "mock"  # from settings.agent_backend
    assert client.delete("/api/agents/orchestrator").status_code == 403


def test_orchestrator_edits_route_to_the_settings_store(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """PATCH /api/agents/orchestrator writes the orchestrator's config to the
    SETTINGS store (it has no agents.json row) and reads it back through the
    synthetic record: backend, model (per the effective backend) and permission
    mode. A backend change clears its active session."""
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend=Backend.CODEX,
        enable_mock_backend=True,
    )
    app = create_app(collector=fake_collector, settings=settings)
    app.state.agents.set_orchestrator_session("codex-session-live")
    client = TestClient(app)

    resp = client.patch(
        "/api/agents/orchestrator",
        json={"backend": "claude", "model": "claude-x", "permission_mode": "auto"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "claude"
    assert body["model"] == "claude-x"  # routed to claude_model (effective backend)
    assert body["permission_mode"] == "auto"
    # Persisted to settings: a fresh read of the record reflects it.
    orch = client.get("/api/agents/orchestrator").json()
    assert (orch["backend"], orch["model"], orch["permission_mode"]) == (
        "claude",
        "claude-x",
        "auto",
    )
    # The backend change cleared the stale (codex) session.
    assert app.state.agents.orchestrator_session_id() is None


def test_orchestrator_edit_forbidden_when_readonly(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent", state_dir=tmp_path, settings_writable=False
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    resp = client.patch("/api/agents/orchestrator", json={"permission_mode": "edit"})
    assert resp.status_code == 403


def test_orchestrator_chat_uses_server_cwd(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A projectless orchestrator chat streams a turn (no project -> server cwd)
    and persists its session id in memory."""
    client = _client_with_project(fake_collector, tmp_path)
    resp = client.post("/api/agents/orchestrator/chat", json={"message": "hi"})
    assert resp.status_code == 200
    assert '"kind":"done"' in resp.text
    _wait_state(client, "orchestrator", "done")
    assert client.get("/api/agents/orchestrator").json()["session_id"] == "mock-session"


def test_agents_backends_endpoint(fake_collector: Collector, tmp_path: Path) -> None:
    """The picker source: available backends with labels + default models. Mock
    appears because the dev flag is on. Not shadowed by /api/agents/{id}."""
    client = _client_with_project(fake_collector, tmp_path)
    resp = client.get("/api/agents/backends")
    assert resp.status_code == 200
    opts = resp.json()
    assert [o["id"] for o in opts] == ["codex", "claude", "opencode", "mock"]
    by_id = {o["id"]: o for o in opts}
    assert by_id["codex"]["label"] == "Codex"
    assert by_id["claude"]["label"] == "Claude"
    assert by_id["opencode"]["label"] == "Opencode"
    assert by_id["codex"]["default_model"] == "gpt-5.5"
    assert by_id["claude"]["default_model"] == "claude-opus-4-8"
    assert by_id["opencode"]["default_model"] == "gemma-4-26B-A4B-it"
    # Each backend carries a model catalog for the picker's autocomplete, with
    # the default among them.
    assert by_id["claude"]["models"] == [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ]
    assert by_id["codex"]["default_model"] in by_id["codex"]["models"]


def test_agents_backends_models_include_env_override(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A claude model set via env that is outside the built-in catalog is still
    offered (the default is prepended), so the picker never hides the effective
    default."""
    client = TestClient(
        create_app(
            collector=fake_collector,
            settings=Settings(
                web_dist=tmp_path / "absent",
                state_dir=tmp_path,
                claude_model="claude-custom-tier",
            ),
        )
    )
    by_id = {o["id"]: o for o in client.get("/api/agents/backends").json()}
    assert by_id["claude"]["default_model"] == "claude-custom-tier"
    assert by_id["claude"]["models"][0] == "claude-custom-tier"


def test_agents_backends_hides_mock_without_flag(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """Without the dev flag, mock is not an offered backend."""
    client = TestClient(
        create_app(
            collector=fake_collector,
            settings=Settings(web_dist=tmp_path / "absent", state_dir=tmp_path),
        )
    )
    ids = [o["id"] for o in client.get("/api/agents/backends").json()]
    assert ids == ["codex", "claude", "opencode"]


def test_patch_backend_redefaults_model_via_api(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """PATCHing the backend without a model re-stamps the model to the new
    backend's default (the reported gpt-5.5-sticks-on-claude bug, over HTTP)."""
    client = _client_with_project(fake_collector, tmp_path)
    client.post(
        "/api/agents",
        json={"name": "Builder", "project_id": "my-app", "backend": "codex"},
    )
    assert client.get("/api/agents/builder").json()["model"] == "gpt-5.5"

    patched = client.patch("/api/agents/builder", json={"backend": "claude"})
    assert patched.status_code == 200
    assert patched.json()["model"] == "claude-opus-4-8"


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
        agent_backend=Backend.MOCK,
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


_client_stack: ExitStack | None = None


@pytest.fixture(autouse=True)
def _hold_clients_open() -> Iterator[None]:
    """Keep every `_agent_client` open for the whole test.

    A `TestClient` used outside a context manager starts and stops a portal per
    request, so a supervised run - which by design outlives the request that
    started it - is cancelled the moment that request returns, and only finishes
    if it happens to win the race. It used to win; offloading the store calls to
    `asyncio.to_thread` added real thread hops to a turn and it stopped. The
    race was the bug, not the timing: this is the same reasoning as
    `conftest.make_client`.
    """
    global _client_stack
    with ExitStack() as stack:
        _client_stack = stack
        yield
        _client_stack = None


def _agent_client(
    fake_collector: Collector, tmp_path: Path, *, goal: str = "do the thing"
) -> TestClient:
    """A mock-backend app with project 'my-app' and agent 'builder' (a goal)."""
    proj = tmp_path / "proj"
    proj.mkdir()
    assert _client_stack is not None, "_hold_clients_open is autouse"
    client = _client_stack.enter_context(
        TestClient(
            create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
        )
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


def test_agent_run_streamerror_persists_error_with_detail(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend that ends a turn with a terminal StreamError (idle timeout,
    over-limit line, thread-setup failure) makes the run complete normally
    (RunPhase DONE), but the agent must persist ERROR with the detail as its
    durable outcome message - so pending_agents surfaces WHY, not an empty error.
    Regression pin for the orchestrator-visibility gap (task 20260727-140443)."""
    from scufris import backends as backends_mod

    async def stream_then_error(
        self: object,
        settings: Settings,
        prompt: str,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamSessionStarted(session_id="mock-session")
        yield StreamError(detail="app-server timed out after 120s")

    monkeypatch.setattr(backends_mod.MockBackend, "stream", stream_then_error)
    client = _agent_client(fake_collector, tmp_path)
    assert client.post("/api/agents/builder/run", json={}).status_code == 200

    # The PERSISTED record carries the terminal state (the persist callback runs in
    # the supervisor's finally, after the run settles). /status reports the live
    # RunPhase (DONE - a StreamError is a normal terminal event, decision on file),
    # so poll the durable agent record instead.
    agent: dict = {}
    for _ in range(200):
        agent = client.get("/api/agents/builder").json()
        if agent.get("state") == "error":
            break
        time.sleep(0.01)
    assert agent["state"] == "error"
    # The failure is surfaced to the orchestrator poll with its diagnostic detail.
    pending = client.get("/api/agents/pending").json()
    row = next(r for r in pending if r["agent_id"] == "builder")
    assert row["state"] == "error"
    assert row["message"] == "app-server timed out after 120s"


def test_agent_run_exception_persists_error_with_detail(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend that RAISES (not a yielded StreamError) ends the run in RunPhase
    ERROR with run.error = str(exc); persist must use that detail as the durable
    outcome message too - the pre-existing exception paths previously persisted an
    empty message. Pins the DECISION.md claim that the fix improves them as well."""
    from scufris import backends as backends_mod

    async def stream_raises(
        self: object,
        settings: Settings,
        prompt: str,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        raise RuntimeError("backend exploded mid-turn")
        yield  # pragma: no cover - marks this an async generator (unreachable)

    monkeypatch.setattr(backends_mod.MockBackend, "stream", stream_raises)
    client = _agent_client(fake_collector, tmp_path)
    assert client.post("/api/agents/builder/run", json={}).status_code == 200

    agent: dict = {}
    for _ in range(200):
        agent = client.get("/api/agents/builder").json()
        if agent.get("state") == "error":
            break
        time.sleep(0.01)
    assert agent["state"] == "error"
    pending = client.get("/api/agents/pending").json()
    row = next(r for r in pending if r["agent_id"] == "builder")
    assert row["message"] == "backend exploded mid-turn"


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


def test_request_input_records_waiting_outcome(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """POST /request_input records a WAITING outcome carrying the question (BC2),
    readable back via the agent's status/outcome. Returns immediately."""
    client = _agent_client(fake_collector, tmp_path)
    resp = client.post(
        "/api/agents/builder/request_input",
        json={"question": "should I merge to master?"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"agent_id": "builder", "state": "waiting"}


def test_request_input_validates_and_404s(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = _agent_client(fake_collector, tmp_path)
    # Empty question -> 422.
    assert (
        client.post(
            "/api/agents/builder/request_input", json={"question": "  "}
        ).status_code
        == 422
    )
    # Unknown agent -> 404.
    assert (
        client.post(
            "/api/agents/ghost/request_input", json={"question": "hi"}
        ).status_code
        == 404
    )
    # The orchestrator is not a sub-agent -> 404 (it resolves but has no row).
    assert (
        client.post(
            "/api/agents/orchestrator/request_input", json={"question": "hi"}
        ).status_code
        == 404
    )


def test_report_back_records_reported_outcome(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """POST /report_back records a REPORTED outcome carrying the summary, readable
    back via the agent's status/outcome. Returns immediately."""
    client = _agent_client(fake_collector, tmp_path)
    resp = client.post(
        "/api/agents/builder/report_back",
        json={"summary": "implemented X; tests green"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"agent_id": "builder", "state": "reported"}


def test_report_back_validates_and_404s(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = _agent_client(fake_collector, tmp_path)
    # Empty summary -> 422.
    assert (
        client.post(
            "/api/agents/builder/report_back", json={"summary": "  "}
        ).status_code
        == 422
    )
    # Unknown agent -> 404.
    assert (
        client.post(
            "/api/agents/ghost/report_back", json={"summary": "done"}
        ).status_code
        == 404
    )
    # The orchestrator is not a sub-agent -> 404 (it resolves but has no row).
    assert (
        client.post(
            "/api/agents/orchestrator/report_back", json={"summary": "done"}
        ).status_code
        == 404
    )


def test_reported_agent_shows_in_pending_and_acknowledges(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A sub-agent that called report_back shows up in /api/agents/pending with
    state=reported and its summary; acknowledging clears it from the poll."""
    client = _agent_client(fake_collector, tmp_path)
    client.post("/api/agents/builder/report_back", json={"summary": "shipped X"})
    pending = client.get("/api/agents/pending").json()
    assert len(pending) == 1
    assert pending[0]["agent_id"] == "builder"
    assert pending[0]["state"] == "reported"
    assert pending[0]["message"] == "shipped X"

    assert client.post("/api/agents/builder/acknowledge").json() == {
        "agent_id": "builder",
        "acknowledged": True,
    }
    assert client.get("/api/agents/pending").json() == []


def test_pending_agents_and_acknowledge_roundtrip(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A sub-agent that called request_input shows up in /api/agents/pending with
    its question; acknowledging it clears it from the poll (BC3). The static
    /pending route is declared before /api/agents/{id}, so it is not shadowed - a
    non-empty result here proves that."""
    client = _agent_client(fake_collector, tmp_path)
    # Nothing pending yet (a list, not the {id} route's "no such agent" 404).
    assert client.get("/api/agents/pending").json() == []

    client.post("/api/agents/builder/request_input", json={"question": "merge?"})
    pending = client.get("/api/agents/pending").json()
    assert len(pending) == 1
    assert pending[0]["agent_id"] == "builder"
    assert pending[0]["state"] == "waiting"
    assert pending[0]["message"] == "merge?"

    ack = client.post("/api/agents/builder/acknowledge")
    assert ack.status_code == 200
    assert ack.json() == {"agent_id": "builder", "acknowledged": True}
    assert client.get("/api/agents/pending").json() == []
    # Idempotent: a second ack (or an unknown agent) reports acknowledged=False.
    assert (
        client.post("/api/agents/builder/acknowledge").json()["acknowledged"] is False
    )
    assert client.post("/api/agents/ghost/acknowledge").json()["acknowledged"] is False


def test_spawn_records_parent_on_child(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launching a child with a parent_session_id records (orchestrator, chat) on
    the child, so a later request_input can be routed back to that chat (part 3)."""
    _use_fake_backend(monkeypatch)
    proj = tmp_path / "proj"
    proj.mkdir()
    app = create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    client = TestClient(app)
    client.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
    client.post(
        "/api/agents",
        json={
            "name": "Builder",
            "project_id": "my-app",
            "backend": "mock",
            "goal": "g",
        },
    )
    resp = client.post("/api/agents/builder/run", json={"parent_session_id": "chat-1"})
    assert resp.status_code == 200
    assert app.state.agents.parent_of("builder") == ("orchestrator", "chat-1")
    # A run with no parent leaves the child unattributed (back-compat).
    client.post(
        "/api/agents",
        json={"name": "Loner", "project_id": "my-app", "backend": "mock", "goal": "g"},
    )
    client.post("/api/agents/loner/run", json={})
    assert app.state.agents.parent_of("loner") == (None, None)


def test_pending_filtered_by_parent_session(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A chat's pending poll returns its own children PLUS unattributed ones, but
    not another chat's - and each row is annotated with its parent (part 3)."""
    app = create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    client = TestClient(app)
    proj = tmp_path / "proj"
    proj.mkdir()
    client.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
    for name in ("Builder", "Helper", "Loner"):
        client.post(
            "/api/agents",
            json={"name": name, "project_id": "my-app", "backend": "mock", "goal": "g"},
        )
    agents = app.state.agents
    agents.record_spawn_parent("builder", "orchestrator", "chat-1")
    agents.record_spawn_parent("helper", "orchestrator", "chat-2")
    # loner is left unattributed (UI-launched).
    for aid in ("builder", "helper", "loner"):
        client.post(f"/api/agents/{aid}/request_input", json={"question": f"{aid}?"})

    # chat-1 sees its own child + the unattributed one, not chat-2's.
    rows = client.get("/api/agents/pending?parent_session_id=chat-1").json()
    assert {r["agent_id"] for r in rows} == {"builder", "loner"}
    builder = next(r for r in rows if r["agent_id"] == "builder")
    assert builder["parent_session_id"] == "chat-1"
    assert builder["parent_agent_id"] == "orchestrator"
    loner = next(r for r in rows if r["agent_id"] == "loner")
    assert loner["parent_session_id"] is None
    # No filter -> all three (back-compat with the single-chat poll).
    assert len(client.get("/api/agents/pending").json()) == 3


def test_agent_turn_threads_its_id_to_the_backend(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regular agent's turn passes its own id (and is_orchestrator=False) to the
    backend, so its scufris server runs in the AGENT role addressed to it - the
    wiring request_input depends on (BC2)."""
    fake = _use_fake_backend(monkeypatch)
    proj = tmp_path / "proj"
    proj.mkdir()
    client = TestClient(
        create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    )
    client.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
    client.post(
        "/api/agents",
        json={"name": "Builder", "project_id": "my-app", "backend": "mock"},
    )
    assert (
        client.post("/api/agents/builder/chat", json={"message": "hi"}).status_code
        == 200
    )
    _wait_state(client, "builder", "done")
    assert fake.agent_id == "builder"
    assert fake.is_orchestrator is False


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


def test_agent_chat_streams_and_persists_session(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A chat turn streams as SSE and persists the (resumed) session id, so the
    next turn continues the same conversation."""
    client = _agent_client(fake_collector, tmp_path)
    resp = client.post("/api/agents/builder/chat", json={"message": "hi there"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    # The mock echoes the prompt as a text delta and finishes with a done frame.
    assert '"kind":"text_delta"' in resp.text
    assert '"kind":"done"' in resp.text
    assert "hi there" in resp.text

    _wait_state(client, "builder", "done")
    agent = client.get("/api/agents/builder").json()
    assert agent["session_id"] == "mock-session"  # captured + persisted

    # A second turn resumes the SAME session (mock returns the id it was given).
    resp2 = client.post("/api/agents/builder/chat", json={"message": "again"})
    assert resp2.status_code == 200
    _wait_state(client, "builder", "done")
    assert client.get("/api/agents/builder").json()["session_id"] == "mock-session"


def test_agent_chat_validates(fake_collector: Collector, tmp_path: Path) -> None:
    client = _agent_client(fake_collector, tmp_path)
    # Empty / whitespace message -> 422.
    assert (
        client.post("/api/agents/builder/chat", json={"message": "  "}).status_code
        == 422
    )
    # Unknown agent -> 404.
    assert (
        client.post("/api/agents/ghost/chat", json={"message": "hi"}).status_code == 404
    )


class _ForkFakeBackend:
    """A backend whose transcript is canned and whose stream records the prompt
    it was asked, so the per-agent revert-fork can be asserted end to end."""

    name = "fake"

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.session_ids: list[str | None] = []

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
        self.prompts.append(prompt)
        self.session_ids.append(session_id)
        # A resumed turn keeps its id; a fresh turn (session_id=None, the revert)
        # opens "sess-new" - so the tail-drop is observable.
        yield StreamDone(
            reply=AgentReply(text=f"reply: {prompt}", status="completed"),
            session_id=session_id or "sess-new",
        )

    def read_status(self, settings: Settings, session_id: str | None) -> None:
        return None

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        # A 3-message conversation to fork from.
        return [
            TranscriptMessage(role="user", text="first question"),
            TranscriptMessage(role="assistant", text="first answer"),
            TranscriptMessage(role="user", text="second question"),
        ]


def test_agent_fork_reverts_single_session(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project agent's fork rewinds its one session to the fork point and
    continues from the edit: the seed carries the prior turns + the edit, the
    turn opens a FRESH session (the old tail is dropped), and it streams SSE."""
    fake = _ForkFakeBackend()
    patch_get_backend(monkeypatch, fake)
    client = _agent_client(fake_collector, tmp_path)
    # Give the agent a session first, so the fork's revert to a new one is visible.
    client.post("/api/agents/builder/chat", json={"message": "seed"})
    _wait_state(client, "builder", "done")
    assert client.get("/api/agents/builder").json()["session_id"] == "sess-new"

    # Fork at the second user message (index 2), editing its text.
    resp = client.post(
        "/api/agents/builder/fork",
        json={"message_index": 2, "text": "edited second"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert '"kind":"done"' in resp.text
    _wait_state(client, "builder", "done")

    # The seed prompt (the fork turn) carries the prior turns + the edit, and drops
    # the message AFTER the fork point (the original "second question").
    seed = fake.prompts[-1]
    assert "first question" in seed
    assert "first answer" in seed
    assert seed.rstrip().endswith("edited second")
    assert "second question" not in seed
    # The fork launched a FRESH session (session_id=None passed), the revert.
    assert fake.session_ids[-1] is None


def test_agent_fork_validates(fake_collector: Collector, tmp_path: Path) -> None:
    client = _client_with_project(fake_collector, tmp_path)
    client.post(
        "/api/agents",
        json={"name": "Builder", "project_id": "my-app", "backend": "mock"},
    )
    # Empty text -> 422; unknown agent -> 404.
    assert (
        client.post(
            "/api/agents/builder/fork", json={"message_index": 0, "text": " "}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/agents/ghost/fork", json={"message_index": 0, "text": "hi"}
        ).status_code
        == 404
    )
    # The orchestrator keeps its own multi-session fork -> 409 here.
    assert (
        client.post(
            "/api/agents/orchestrator/fork", json={"message_index": 0, "text": "hi"}
        ).status_code
        == 409
    )
    # A project agent whose project was deleted -> 422 (missing project), the
    # docstring-advertised branch: orphan the agent, then fork with real text.
    assert client.delete("/api/projects/my-app").status_code == 200
    assert (
        client.post(
            "/api/agents/builder/fork", json={"message_index": 0, "text": "hi"}
        ).status_code
        == 422
    )


def test_project_lookup_never_runs_on_the_event_loop(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The async agent routes offload their project lookup to a worker thread.

    `ProjectStore.get` is a synchronous `Database.transaction()`, and every begin
    on that engine is `BEGIN IMMEDIATE` - so it takes SQLite's single write lock
    and waits up to `busy_timeout` for it. Run on the loop thread it stalls every
    other request, stream and probe in the process; measured at 3.04s against a
    0.01s heartbeat before this was offloaded. The invariant is exactly "no
    running loop in the thread that opens the transaction", so that is what is
    asserted, rather than a timing.
    """
    lookups: list[bool] = []
    original = ProjectStore.get

    def recording_get(self: ProjectStore, project_id: str) -> Project:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            lookups.append(False)
        else:
            lookups.append(True)
        return original(self, project_id)

    fake = _ForkFakeBackend()
    patch_get_backend(monkeypatch, fake)
    monkeypatch.setattr(ProjectStore, "get", recording_get)
    client = _agent_client(fake_collector, tmp_path)

    # All three async routes that resolve an agent's project.
    assert client.post("/api/agents/builder/run", json={}).status_code == 200
    _wait_state(client, "builder", "done")
    assert client.post("/api/agents/builder/chat", json={"message": "hi"}).status_code
    _wait_state(client, "builder", "done")
    assert client.post(
        "/api/agents/builder/fork", json={"message_index": 0, "text": "edited"}
    ).status_code
    _wait_state(client, "builder", "done")

    # The provoking stimulus really fired: the routes did look a project up.
    assert len(lookups) >= 3
    assert True not in lookups


async def test_agent_chat_conflicts_with_active_run(
    fake_collector: Collector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chat turn goes through the same per-agent supervisor slot as a run, so a
    second turn while one is in flight is refused with 409. Driven async: both
    the TestClient and ASGITransport read the whole SSE body before returning, so
    a held-open turn needs concurrent requests on one loop."""
    import httpx

    from scufris import backends as backends_mod

    release = asyncio.Event()

    async def blocking_stream(
        self: object,
        settings: Settings,
        prompt: str,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamTextDelta(delta="working")
        await release.wait()  # hold the run active until the test releases it
        yield StreamDone(reply=AgentReply(text="done"), session_id="mock-session")

    monkeypatch.setattr(backends_mod.MockBackend, "stream", blocking_stream)
    proj = tmp_path / "proj"
    proj.mkdir()
    app = await asyncio.to_thread(
        create_app, collector=fake_collector, settings=_mock_settings(tmp_path)
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            await ac.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
            await ac.post(
                "/api/agents",
                json={
                    "name": "Builder",
                    "project_id": "my-app",
                    "backend": "mock",
                    "goal": "g",
                },
            )
            # The first turn buffers its whole SSE body, so it stays pending
            # (blocked on `release`); its run registers before it yields.
            first = asyncio.create_task(
                ac.post("/api/agents/builder/chat", json={"message": "one"})
            )
            for _ in range(200):
                await asyncio.sleep(0.01)
                st = (await ac.get("/api/agents/builder/status")).json()
                if st["state"] in ("queued", "running"):
                    break
            # A second turn while the first is active -> 409.
            second = await ac.post("/api/agents/builder/chat", json={"message": "two"})
            assert second.status_code == 409
            release.set()
            r1 = await first
            assert r1.status_code == 200
    finally:
        release.set()


async def test_simultaneous_agent_chats_start_exactly_one_run(
    fake_collector: Collector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eight chat turns fired at once for ONE agent start exactly one run.

    The companion of `test_agent_chat_conflicts_with_active_run`, which polls
    `/status` first and so only ever enters the guard once the run is registered.
    This one fires without polling, which is the window `AgentRunService.launch`
    becoming a coroutine opened: the check reads `supervisor.status`, which is
    None until `supervisor.start` registers the run, and the `mark_running`
    offload now yields between the two. Two turns passing the check both write
    `agent_runs[agent_id]`, and the survivor is the only one `cancel_agent_run`
    can stop - so a live run becomes unstoppable (review round 1, R1.1).

    `mark_running` is slowed by 50ms so the window is ENTERED rather than won:
    the defect is that the window exists at all, and a test that has to beat a
    real store write to it reproduces it only some of the time. The delay runs in
    the worker thread the offload already uses, so it changes the timing of the
    thing under test and nothing else."""
    import httpx

    from scufris import backends as backends_mod

    release = asyncio.Event()
    real_mark_running = AgentStore.mark_running

    def slow_mark_running(self: AgentStore, agent_id: str) -> AgentRecord:
        time.sleep(0.05)
        return real_mark_running(self, agent_id)

    monkeypatch.setattr(AgentStore, "mark_running", slow_mark_running)

    async def blocking_stream(
        self: object,
        settings: Settings,
        prompt: str,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamTextDelta(delta="working")
        await release.wait()
        yield StreamDone(reply=AgentReply(text="done"), session_id="mock-session")

    monkeypatch.setattr(backends_mod.MockBackend, "stream", blocking_stream)
    proj = tmp_path / "proj"
    proj.mkdir()
    app = await asyncio.to_thread(
        create_app, collector=fake_collector, settings=_mock_settings(tmp_path)
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            await ac.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
            await ac.post(
                "/api/agents",
                json={
                    "name": "Builder",
                    "project_id": "my-app",
                    "backend": "mock",
                    "goal": "g",
                },
            )
            turns = [
                asyncio.create_task(
                    ac.post("/api/agents/builder/chat", json={"message": f"m{i}"})
                )
                for i in range(8)
            ]
            # No /status poll: the point is to enter the guard BEFORE the run is
            # registered. The winner stays blocked on `release`, so waiting for
            # seven turns to come back is waiting for seven refusals - and when
            # the guard leaks, they never come back and the timeout below leaves
            # the extra 200s to assert on.
            for _ in range(500):
                await asyncio.sleep(0.01)
                if sum(t.done() for t in turns) >= 7:
                    break
            release.set()
            statuses = sorted(r.status_code for r in await asyncio.gather(*turns))
        assert statuses == [200] + [409] * 7, statuses
    finally:
        release.set()


async def _wait_running(ac: object, path: str) -> None:
    """Poll an agent's /status until its run is queued/running (the blocking
    backend has registered the run), or give up after ~2s."""
    for _ in range(200):
        await asyncio.sleep(0.01)
        st = (await ac.get(path)).json()  # type: ignore[attr-defined]
        if st["state"] in ("queued", "running"):
            return


async def _wait_outcome(store: object, agent_id: str, state: str) -> str | None:
    """Poll the durable outcome store until the agent reaches ``state`` (the
    persist callback runs in the supervisor's finally, after the relay ends).

    Off-loop like every other store read from an `async def`: the read opens a
    transaction, and `Database.transaction()` refuses the loop thread."""
    for _ in range(200):
        outcome = await asyncio.to_thread(store.outcome, agent_id)  # type: ignore[attr-defined]
        if outcome is not None and outcome.state == state:
            return outcome.state
        await asyncio.sleep(0.01)
    outcome = await asyncio.to_thread(store.outcome, agent_id)  # type: ignore[attr-defined]
    return outcome.state if outcome is not None else None


async def test_cancel_endpoint_stops_run_and_marks_cancelled(
    fake_collector: Collector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/agents/{id}/cancel stops a live run: 200 cancelled=true, the held
    turn ends, and the agent's DURABLE outcome is CANCELLED (not ERROR) - so a
    user stop reads as neutral and does NOT surface in pending_agents. Asserted on
    the durable record, not /status (lesson assert-terminal-outcome-on-the-durable-
    record-not-status)."""
    import httpx

    from scufris import backends as backends_mod

    release = asyncio.Event()

    async def blocking_stream(
        self: object, settings: Settings, prompt: str, **kwargs: object
    ) -> AsyncIterator[StreamEvent]:
        yield StreamTextDelta(delta="working")
        await release.wait()  # hold the run active until cancelled/released
        yield StreamDone(reply=AgentReply(text="done"), session_id="mock-session")

    monkeypatch.setattr(backends_mod.MockBackend, "stream", blocking_stream)
    proj = tmp_path / "proj"
    proj.mkdir()
    app = await asyncio.to_thread(
        create_app, collector=fake_collector, settings=_mock_settings(tmp_path)
    )
    store = app.state.agents
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            await ac.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
            await ac.post(
                "/api/agents",
                json={
                    "name": "Builder",
                    "project_id": "my-app",
                    "backend": "mock",
                    "goal": "g",
                },
            )
            turn = asyncio.create_task(
                ac.post("/api/agents/builder/chat", json={"message": "one"})
            )
            await _wait_running(ac, "/api/agents/builder/status")

            resp = await ac.post("/api/agents/builder/cancel")
            assert resp.status_code == 200
            assert resp.json() == {"agent_id": "builder", "cancelled": True}

            r1 = await turn  # the cancel ends the held turn's SSE relay
            assert r1.status_code == 200
            assert (
                await _wait_outcome(store, "builder", AgentState.CANCELLED)
                == AgentState.CANCELLED
            )
            # A user-cancelled agent is NOT pending to the orchestrator.
            assert "builder" not in await asyncio.to_thread(store.pending_outcomes)
            # Cancelling again (nothing live) -> 404.
            assert (await ac.post("/api/agents/builder/cancel")).status_code == 404
    finally:
        release.set()


async def test_cancel_endpoint_404_when_idle_or_unknown(
    fake_collector: Collector,
    tmp_path: Path,
) -> None:
    """404 when the agent exists but has no active run, and 404 for an unknown
    agent - there is nothing to cancel in either case."""
    import httpx

    proj = tmp_path / "proj"
    proj.mkdir()
    app = await asyncio.to_thread(
        create_app, collector=fake_collector, settings=_mock_settings(tmp_path)
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        await ac.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
        await ac.post(
            "/api/agents",
            json={"name": "Builder", "project_id": "my-app", "backend": "mock"},
        )
        assert (await ac.post("/api/agents/builder/cancel")).status_code == 404
        assert (await ac.post("/api/agents/ghost/cancel")).status_code == 404


async def test_cancel_orchestrator_run(
    fake_collector: Collector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator landing chat is cancellable through the SAME per-agent
    endpoint via ORCHESTRATOR_ID: it is an agent in the run registry, so hitting
    /api/agents/orchestrator/cancel stops its turn and marks it CANCELLED."""
    import httpx

    from scufris import backends as backends_mod

    release = asyncio.Event()

    async def blocking_stream(
        self: object, settings: Settings, prompt: str, **kwargs: object
    ) -> AsyncIterator[StreamEvent]:
        yield StreamTextDelta(delta="working")
        await release.wait()
        yield StreamDone(reply=AgentReply(text="done"), session_id="sess-live")

    monkeypatch.setattr(backends_mod.MockBackend, "stream", blocking_stream)
    app = await asyncio.to_thread(
        create_app, collector=fake_collector, settings=_mock_settings(tmp_path)
    )
    store = app.state.agents
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            turn = asyncio.create_task(
                ac.post("/api/chat/stream", json={"message": "hi"})
            )
            await _wait_running(ac, "/api/agents/orchestrator/status")

            resp = await ac.post("/api/agents/orchestrator/cancel")
            assert resp.status_code == 200
            assert resp.json()["cancelled"] is True

            r = await turn
            assert r.status_code == 200
            assert (
                await _wait_outcome(store, "orchestrator", AgentState.CANCELLED)
                == AgentState.CANCELLED
            )
    finally:
        release.set()


async def test_orchestrator_session_recorded_at_turn_start(
    fake_collector: Collector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator's session id is recorded in the registry the moment the
    backend emits ``StreamSessionStarted`` (turn-start), NOT only at mark_finished -
    so a client refreshing mid-turn sees the session. Once the turn settles the id
    persists, with a single history entry (the early record + terminal record are
    idempotent)."""
    import httpx

    from scufris import backends as backends_mod

    release = asyncio.Event()

    async def session_first_stream(
        self: object,
        settings: Settings,
        prompt: str,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamSessionStarted(session_id="sess-live")
        yield StreamTextDelta(delta="working")
        await release.wait()  # hold the run active so we can observe mid-turn
        yield StreamDone(reply=AgentReply(text="done"), session_id="sess-live")

    monkeypatch.setattr(backends_mod.MockBackend, "stream", session_first_stream)
    app = await asyncio.to_thread(
        create_app, collector=fake_collector, settings=_mock_settings(tmp_path)
    )
    store = app.state.agents
    # Fresh chat: no session recorded before the turn.
    assert await asyncio.to_thread(store.orchestrator_session_id) is None
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            turn = asyncio.create_task(
                ac.post("/api/chat/stream", json={"message": "hi"})
            )
            # The session id is recorded WHILE the turn is still streaming.
            current: str | None = None
            for _ in range(200):
                await asyncio.sleep(0.01)
                current = await asyncio.to_thread(store.orchestrator_session_id)
                if current is not None:
                    break
            assert current == "sess-live"
            assert not release.is_set()  # proven mid-turn, before the done frame
            release.set()
            r = await turn
            assert r.status_code == 200
            # Persists after settle, and the early + terminal records did not
            # double-append to the switcher history.
            assert await asyncio.to_thread(store.orchestrator_session_id) == "sess-live"
            assert await asyncio.to_thread(store.orchestrator_sessions) == ["sess-live"]
    finally:
        release.set()


async def test_orchestrator_session_recorded_even_when_turn_errors(
    fake_collector: Collector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn that starts its thread (StreamSessionStarted) then ERRORS before any
    done frame still leaves the session recorded - the thread exists on disk, so
    ownership is kept, consistent with mark_finished-on-error."""
    from scufris import backends as backends_mod

    async def session_then_error(
        self: object,
        settings: Settings,
        prompt: str,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamSessionStarted(session_id="sess-doomed")
        yield StreamError(detail="app-server blew up")

    monkeypatch.setattr(backends_mod.MockBackend, "stream", session_then_error)
    app = await asyncio.to_thread(
        create_app, collector=fake_collector, settings=_mock_settings(tmp_path)
    )
    store = app.state.agents
    resp = TestClient(app).post("/api/chat/stream", json={"message": "hi"})
    assert resp.status_code == 200
    assert '"kind":"error"' in resp.text
    # The session is still owned by the orchestrator despite the failed turn.
    assert await asyncio.to_thread(store.orchestrator_session_id) == "sess-doomed"
    assert await asyncio.to_thread(store.orchestrator_sessions) == ["sess-doomed"]


async def test_status_exposes_in_flight_prompt_stripped(
    fake_collector: Collector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/status` carries the in-flight turn's prompt (steering stripped) while the
    run is live, so a client reattaching mid-turn can render the user bubble the
    rollout has not yet been flushed with. Once the run settles it is None again.
    Mirrors the concurrent-turn harness: the turn buffers its whole SSE body, so
    it stays pending on `release` while we poll `/status`."""
    import httpx

    from scufris import backends as backends_mod

    release = asyncio.Event()

    async def blocking_stream(
        self: object,
        settings: Settings,
        prompt: str,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamTextDelta(delta="working")
        await release.wait()  # hold the run active until the test releases it
        yield StreamDone(reply=AgentReply(text="done"), session_id="mock-session")

    monkeypatch.setattr(backends_mod.MockBackend, "stream", blocking_stream)
    proj = tmp_path / "proj"
    proj.mkdir()
    app = await asyncio.to_thread(
        create_app, collector=fake_collector, settings=_mock_settings(tmp_path)
    )
    transport = httpx.ASGITransport(app=app)
    # A message that ALREADY carries the steering block, so the endpoint's
    # strip_steering transform is exercised end to end (matches read_transcript).
    steered = f"{STEERING_PREAMBLE}\n\nwhat is using the most memory?"
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            await ac.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
            await ac.post(
                "/api/agents",
                json={
                    "name": "Builder",
                    "project_id": "my-app",
                    "backend": "mock",
                    "goal": "g",
                },
            )
            turn = asyncio.create_task(
                ac.post("/api/agents/builder/chat", json={"message": steered})
            )
            live: dict[str, object] = {}
            for _ in range(200):
                await asyncio.sleep(0.01)
                live = (await ac.get("/api/agents/builder/status")).json()
                if live["state"] in ("queued", "running"):
                    break
            # The prompt is exposed while live, with the steering block removed.
            assert live["state"] in ("queued", "running")
            assert live["prompt"] == "what is using the most memory?"
            release.set()
            r = await turn
            assert r.status_code == 200
            # Once the run has settled the in-flight prompt is gone again.
            done = (await ac.get("/api/agents/builder/status")).json()
            assert done["prompt"] is None
    finally:
        release.set()


def test_agent_transcript_empty_for_unrun_agent(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = _agent_client(fake_collector, tmp_path)
    resp = client.get("/api/agents/builder/transcript")
    assert resp.status_code == 200
    assert resp.json() == {"messages": []}
    # Unknown agent -> 404.
    assert client.get("/api/agents/ghost/transcript").status_code == 404


def test_agent_transcript_reads_claude_session(
    fake_collector: Collector, tmp_path: Path, database: Database
) -> None:
    """The transcript endpoint reads the agent's backend session history - here a
    real claude session JSONL, seeded on disk and bound to the agent."""
    claude_home = tmp_path / "claude"
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        claude_home=claude_home,
        enable_mock_backend=True,
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    projects = ProjectStore(settings, database)
    projects.create(name="My App", cwd=str(proj))
    store = AgentStore(settings, projects, database)
    store.create(name="Builder", project_id="my-app", backend="claude")
    # Bind a session id to the agent (as a finished turn would).
    store.mark_finished("builder", state=AgentState.DONE, session_id="sess-1")

    sess_dir = claude_home / "projects" / "x"
    sess_dir.mkdir(parents=True)
    (sess_dir / "sess-1.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "message": {"content": "hello"}}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": "hi there"}],
                            "usage": {"input_tokens": 5, "output_tokens": 2},
                        },
                    }
                ),
            ]
        )
    )

    client = TestClient(create_app(collector=fake_collector, settings=settings))
    msgs = client.get("/api/agents/builder/transcript").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["text"] == "hello"
    assert msgs[1]["text"] == "hi there"
    assert msgs[1]["usage"]["output_tokens"] == 2


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
        "auth",
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

    assert tag_of("/api/auth/login", "post") == ["auth"]
    assert tag_of("/api/stats") == ["host"]
    assert tag_of("/api/config") == ["app"]
    assert tag_of("/api/chat/stream", "post") == ["chat"]
    assert tag_of("/api/agent/info") == ["chat"]  # chat, not settings
    assert tag_of("/api/agent/sessions") == ["sessions"]
    assert tag_of("/api/agent/config") == ["settings"]
    assert tag_of("/api/projects", "post") == ["projects"]
    assert tag_of("/api/agents", "get") == ["agents"]  # plural, not settings
    assert tag_of("/api/agents/{agent_id}/run", "post") == ["agents"]
    assert tag_of("/api/agents/pending") == ["agents"]  # the poll, agents tag

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
        agent_backend=Backend.MOCK,
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


def test_project_detail_page_serves_shell(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """/projects/<id> serves the project-detail SPA shell; /api/projects/<id> is
    unaffected; /projects/ (list) is not shadowed by the detail route."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "project-detail.html").write_text("<html>PROJECT SHELL</html>")
    (dist / "index.html").write_text("<html>landing</html>")
    (dist / "projects").mkdir()
    (dist / "projects" / "index.html").write_text("<html>PROJECTS LIST</html>")
    settings = Settings(web_dist=dist, state_dir=tmp_path)
    client = TestClient(create_app(collector=fake_collector, settings=settings))

    # A specific project path -> the detail shell.
    detail = client.get("/projects/my-app")
    assert detail.status_code == 200
    assert "PROJECT SHELL" in detail.text
    # A sub-path -> the same shell.
    assert "PROJECT SHELL" in client.get("/projects/my-app/whatever").text
    # The list path -> the static projects index, NOT the detail shell.
    assert "PROJECTS LIST" in client.get("/projects/").text
    # The JSON API for a project is unaffected (404 unknown id, not the shell).
    api = client.get("/api/projects/my-app")
    assert api.status_code == 404
    assert "PROJECT SHELL" not in api.text


def test_project_detail_page_404_without_frontend(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(web_dist=tmp_path / "absent", state_dir=tmp_path)
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    assert client.get("/projects/my-app").status_code == 404


# --- operator tool console: own-port base + off-loop run (20260723-141026) -----


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_ensure_api_base_defaults_and_respects_override() -> None:
    """ensure_api_base defaults SCUFRIS_API_BASE to the dashboard's OWN port (so
    an in-process tool run loops back to this server, not the :8000 default); an
    explicit override wins. It mutates os.environ directly (as at startup), which
    monkeypatch does not track through setdefault - snapshot/restore explicitly or
    it leaks the base into later respx tests."""
    saved = os.environ.pop("SCUFRIS_API_BASE", None)
    try:
        assert ensure_api_base(Settings(port=7000)) == "http://127.0.0.1:7000"
        assert os.environ["SCUFRIS_API_BASE"] == "http://127.0.0.1:7000"
        # An explicit value wins (setdefault is a no-op over it).
        os.environ["SCUFRIS_API_BASE"] = "http://ops.example:9000"
        assert ensure_api_base(Settings(port=7000)) == "http://ops.example:9000"
    finally:
        if saved is None:
            os.environ.pop("SCUFRIS_API_BASE", None)
        else:
            os.environ["SCUFRIS_API_BASE"] = saved


def test_tool_console_self_loopback(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/agent/tools/pending_agents/run reaches THIS server and returns the
    empty-pending result WITHOUT hanging - proving both the off-loop tool run (the
    tool's BLOCKING httpx would otherwise hang the event loop on self-loopback) and
    the base pointing at the dashboard's own port. Needs a REAL uvicorn socket;
    respx / ASGITransport cannot exercise self-loopback."""
    import httpx
    import uvicorn

    port = _free_port()
    monkeypatch.setenv("SCUFRIS_API_BASE", f"http://127.0.0.1:{port}")
    app = create_app(collector=fake_collector, settings=_mock_settings(tmp_path))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started, "uvicorn did not start"
        # timeout well under the tool's own 15s httpx bound: WITHOUT the off-loop
        # fix this POST hangs (the server loop is blocked serving the loopback) and
        # times out here; WITH it, the callback is served and this returns fast.
        resp = httpx.post(
            f"http://127.0.0.1:{port}/api/agent/tools/pending_agents/run",
            json={"args": {}},
            timeout=8,
        )
        assert resp.status_code == 200
        assert "no agents are waiting" in resp.json()["text"]
    finally:
        server.should_exit = True
        thread.join(timeout=5)


async def test_auto_wake_launches_orchestrator_on_subagent_waiting(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """END-TO-END wiring (BC4): with auto_wake ON, a sub-agent whose in-flight run
    ends with a WAITING outcome (from request_input) causes the orchestrator to be
    GRANTED a turn carrying the question - proving persist -> WakeBridge ->
    AgentRunService.launch. Driven async with a blocked backend so the sub-agent
    run is in-flight when request_input lands; respx/TestClient cannot hold a run
    open."""
    import httpx

    from scufris import backends as backends_mod

    release = asyncio.Event()
    prompts: list[str] = []

    async def blocking_stream(
        self: object, settings: Settings, prompt: str, **kwargs: object
    ) -> AsyncIterator[StreamEvent]:
        prompts.append(prompt)
        yield StreamTextDelta(delta="working")
        await release.wait()  # hold the run in-flight until released
        yield StreamDone(reply=AgentReply(text="done"), session_id="mock-session")

    monkeypatch.setattr(backends_mod.MockBackend, "stream", blocking_stream)

    proj = tmp_path / "proj"
    proj.mkdir()
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend=Backend.MOCK,
        enable_mock_backend=True,
        auto_wake=True,
    )
    app = await asyncio.to_thread(
        create_app, collector=fake_collector, settings=settings
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            await ac.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
            await ac.post(
                "/api/agents",
                json={
                    "name": "Builder",
                    "project_id": "my-app",
                    "backend": "mock",
                    "goal": "do it",
                },
            )
            # Launch the sub-agent run (returns immediately); it blocks on `release`.
            started = await ac.post("/api/agents/builder/run", json={})
            assert started.status_code == 200
            for _ in range(100):
                st = (await ac.get("/api/agents/builder/status")).json()
                if st["state"] == "running":
                    break
                await asyncio.sleep(0.02)
            # The sub-agent asks for a decision mid-run -> WAITING outcome.
            r = await ac.post(
                "/api/agents/builder/request_input",
                json={"question": "merge to master?"},
            )
            assert r.status_code == 200
            # Release -> the run completes WAITING -> the bridge wakes the orchestrator.
            release.set()
            wake = None
            for _ in range(200):
                wake = next(
                    (p for p in prompts if "builder" in p and "merge to master?" in p),
                    None,
                )
                if wake is not None:
                    break
                await asyncio.sleep(0.02)
            assert wake is not None, f"orchestrator was not woken; prompts={prompts}"
            assert "[wake]" in wake
    finally:
        release.set()


async def test_auto_wake_off_does_not_launch_orchestrator(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With auto_wake OFF, a sub-agent finishing WAITING does NOT grant the
    orchestrator a turn (poll-only mode)."""
    import httpx

    from scufris import backends as backends_mod

    release = asyncio.Event()
    prompts: list[str] = []

    async def blocking_stream(
        self: object, settings: Settings, prompt: str, **kwargs: object
    ) -> AsyncIterator[StreamEvent]:
        prompts.append(prompt)
        yield StreamTextDelta(delta="working")
        await release.wait()
        yield StreamDone(reply=AgentReply(text="done"), session_id="mock-session")

    monkeypatch.setattr(backends_mod.MockBackend, "stream", blocking_stream)
    proj = tmp_path / "proj"
    proj.mkdir()
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend=Backend.MOCK,
        enable_mock_backend=True,
        auto_wake=False,
    )
    app = await asyncio.to_thread(
        create_app, collector=fake_collector, settings=settings
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            await ac.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
            await ac.post(
                "/api/agents",
                json={
                    "name": "Builder",
                    "project_id": "my-app",
                    "backend": "mock",
                    "goal": "do it",
                },
            )
            await ac.post("/api/agents/builder/run", json={})
            for _ in range(100):
                st = (await ac.get("/api/agents/builder/status")).json()
                if st["state"] == "running":
                    break
                await asyncio.sleep(0.02)
            await ac.post(
                "/api/agents/builder/request_input", json={"question": "merge?"}
            )
            release.set()
            # Let the run finish; then confirm NO orchestrator run was launched.
            for _ in range(100):
                st = (await ac.get("/api/agents/builder/status")).json()
                if st["state"] == "done":
                    break
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.1)
            orch = await ac.get("/api/agents/orchestrator/status")
            assert orch.json()["state"] != "running"
            assert not any("[wake]" in p for p in prompts)
    finally:
        release.set()


@pytest.mark.parametrize("auto_wake", [True, False])
async def test_stalled_merge_loop_self_heals(
    fake_collector: Collector,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    auto_wake: bool,
) -> None:
    """ACCEPTANCE (BC5): the whole stalled-merge loop self-heals against a faked
    backend, on BOTH wake paths. A sub-agent blocks mid-run (request_input ->
    WAITING); the orchestrator is GRANTED a turn carrying the question (auto_wake
    bridge) or FINDS it by polling pending_agents (auto_wake off); it answers by
    resuming the sub-agent's session (message_agent -> /chat) so the sub-agent
    proceeds to done; and the signal clears. The mock backend runs no real MCP
    tools, so each tool is stood in by the endpoint it calls - the contract under
    test. examples/comms_loop.py is the human-readable walkthrough of the poll
    path; this adds the bridge path and the in-flight run."""
    import httpx

    from scufris import backends as backends_mod

    release = asyncio.Event()
    turns: list[tuple[str, bool, str]] = []  # (agent_id, is_orchestrator, prompt)
    blocked_once = {"done": False}

    async def scripted_stream(
        self: object, settings: Settings, prompt: str, **kwargs: object
    ) -> AsyncIterator[StreamEvent]:
        agent_id = str(kwargs.get("agent_id") or "")
        is_orch = bool(kwargs.get("is_orchestrator"))
        session_id = kwargs.get("session_id")
        turns.append((agent_id, is_orch, prompt))
        # Hold ONLY the sub-agent's first turn in-flight, so request_input lands
        # mid-run; the wake turn and the resume turn each complete at once.
        if agent_id and not is_orch and not blocked_once["done"]:
            blocked_once["done"] = True
            await release.wait()
        yield StreamTextDelta(delta="working")
        yield StreamDone(
            reply=AgentReply(text=f"reply: {prompt}"),
            session_id=session_id if isinstance(session_id, str) else "mock-session",
        )

    monkeypatch.setattr(backends_mod.MockBackend, "stream", scripted_stream)

    proj = tmp_path / "proj"
    proj.mkdir()
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend=Backend.MOCK,
        enable_mock_backend=True,
        auto_wake=auto_wake,
    )
    app = await asyncio.to_thread(
        create_app, collector=fake_collector, settings=settings
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            await ac.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
            await ac.post(
                "/api/agents",
                json={
                    "name": "Builder",
                    "project_id": "my-app",
                    "backend": "mock",
                    "goal": "ship it",
                },
            )
            # 1. The sub-agent run goes in-flight (blocks on `release`).
            started = await ac.post("/api/agents/builder/run", json={})
            assert started.status_code == 200
            for _ in range(200):
                st = (await ac.get("/api/agents/builder/status")).json()
                if st["state"] == "running":
                    break
                await asyncio.sleep(0.02)
            # 2. The sub-agent signals it is blocked mid-run -> WAITING outcome.
            r = await ac.post(
                "/api/agents/builder/request_input",
                json={"question": "merge to master?"},
            )
            assert r.status_code == 200 and r.json()["state"] == "waiting"
            # 3. Release -> the run completes with the WAITING outcome preserved.
            release.set()

            # 4. The orchestrator discovers the blocked sub-agent, by its path.
            if auto_wake:
                # The bridge grants the orchestrator a turn carrying the question.
                wake: str | None = None
                for _ in range(200):
                    wake = next(
                        (
                            p
                            for (aid, orch, p) in turns
                            if orch and "builder" in p and "merge to master?" in p
                        ),
                        None,
                    )
                    if wake is not None:
                        break
                    await asyncio.sleep(0.02)
                assert wake is not None, f"orchestrator was not woken; turns={turns}"
                assert "[wake]" in wake
            else:
                # Poll-only: the orchestrator finds it via pending_agents; no wake.
                pending: list = []
                for _ in range(200):
                    pending = (await ac.get("/api/agents/pending")).json()
                    if pending:
                        break
                    await asyncio.sleep(0.02)
                assert [p["agent_id"] for p in pending] == ["builder"]
                assert pending[0]["message"] == "merge to master?"
                assert not any(orch for (aid, orch, p) in turns), (
                    "poll path must not wake"
                )

            # 5. The orchestrator answers by resuming the sub-agent's own session.
            chat = await ac.post(
                "/api/agents/builder/chat", json={"message": "yes, merge it"}
            )
            assert chat.status_code == 200
            # The sub-agent proceeded: its resume turn ran with the answer, to done.
            for _ in range(200):
                st = (await ac.get("/api/agents/builder/status")).json()
                if st["state"] == "done":
                    break
                await asyncio.sleep(0.02)
            assert st["state"] == "done"
            resumed = [
                p
                for (aid, orch, p) in turns
                if aid == "builder" and not orch and "yes, merge it" in p
            ]
            assert resumed, f"sub-agent did not resume with the answer; turns={turns}"

            # 6. The signal is cleared: the DONE resume (a new run) overwrote the
            #    WAITING outcome, so the loop is resolved; acknowledge is an
            #    idempotent belt-and-suspenders clear.
            await ac.post("/api/agents/builder/acknowledge")
            assert (await ac.get("/api/agents/pending")).json() == []
    finally:
        release.set()


async def test_agent_routes_do_not_stall_the_event_loop(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The routes stay responsive while another writer holds the write lock.

    20260801-120412 measured the opposite for the project routes: an `async def`
    route reached the store directly, so its `BEGIN IMMEDIATE` took SQLite's
    single write lock ON the loop and stalled a 0.01s heartbeat for 3.04s. The
    guard in `Database.transaction()` makes that shape impossible now, and this
    is the behavioural half of the same claim - that the offload the guard forces
    actually keeps the loop free.

    The lock is held from a plain worker thread for longer than a route would
    take on its own, so a route that waited on the loop could not pass by luck.
    The holder is still holding WHILE the four routes are driven, and the ticks
    asserted on are only those counted across the route window - counting the
    whole test, or releasing the lock first, made this pass with the loop free
    (review round 1, R1.2). The holder releases on its own timeout rather than on
    a signal from the routes, because every route below blocks on the same lock:
    a holder waiting for them would deadlock.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    import httpx

    app = await asyncio.to_thread(
        create_app, collector=fake_collector, settings=_mock_settings(tmp_path)
    )
    transport = httpx.ASGITransport(app=app)
    ticks = 0
    stop = asyncio.Event()
    hold = 0.5

    async def heartbeat() -> None:
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    holding = threading.Event()
    release = threading.Event()

    def hold_the_write_lock() -> None:
        with state_database(tmp_path).transaction() as conn:
            conn.execute(sql_text("SELECT 1"))
            holding.set()
            release.wait(timeout=hold)

    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        await ac.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
        await ac.post(
            "/api/agents",
            json={"name": "Builder", "project_id": "my-app", "backend": "mock"},
        )
        beat = asyncio.create_task(heartbeat())
        locker = asyncio.create_task(asyncio.to_thread(hold_the_write_lock))
        await asyncio.to_thread(holding.wait, 10)
        before = ticks
        started = asyncio.get_running_loop().time()
        # Every surface this task moved, driven while the lock IS held.
        statuses = [
            (await ac.get("/api/agents")).status_code,
            (await ac.get("/api/agents/builder")).status_code,
            (await ac.get("/api/agent/sessions")).status_code,
            (
                await ac.patch("/api/agent/config", json={"poll_seconds": 3.0})
            ).status_code,
        ]
        window = asyncio.get_running_loop().time() - started
        during = ticks - before
        release.set()
        stop.set()
        await beat
        await locker

    assert statuses == [200, 200, 200, 200]
    # Every begin is immediate, so all four routes - reads included - waited on
    # the held write lock. If they had waited ON THE LOOP the heartbeat could not
    # have run while they did.
    assert window >= hold * 0.8, f"routes returned in {window:.3f}s, lock held {hold}s"
    assert during > window / 0.02 * 0.5, f"only {during} heartbeats in {window:.3f}s"
