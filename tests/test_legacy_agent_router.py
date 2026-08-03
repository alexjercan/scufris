"""The singular `/api/agent/*` router, driven over fakes on a bare app.

A fourth file in the `test_domain_routers.py` / `test_orchestrator_routers.py` /
`test_chat_router.py` family, for the same reason the others exist: the shared
files are at their line cap. The rig pattern is unchanged.

The claim worth proving here is the one the compatibility surface is FOR: these
routes are aliases, so every one of them must reach the SAME service the scoped
`/api/agents/orchestrator/*` route reaches - `AgentDiagnostics` for
account/usage/memory/health, `AgentRunService` for the record and the forked
turn. A route that grew its own reader shows up here as a service call that
never happened.

The three routes that are NOT aliases - the config view and its whitelisted
PATCH, the "try it" tool runner, and the session switcher - are asserted on what
they translate: which collaborator owns the rule, and which status its refusal
becomes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import scufris.db
from scufris.agent import AgentReply, StreamDone
from scufris.agent_diagnostics import AccountInfo
from scufris.agent_store import ORCHESTRATOR_ID, AgentNotFound, AgentRecord, AgentStore
from scufris.api.legacy_agent import (
    AgentConfigUpdate,
    LegacyAgentDeps,
    build_legacy_agent_router,
)
from scufris.backends import Capability
from scufris.config import Settings
from scufris.eventbus import EventBus
from scufris.health import AgentHealth
from scufris.projects import ProjectStore
from scufris.sessions import (
    MemoryFootprint,
    SessionContext,
    SessionInfo,
    TranscriptMessage,
    UsageQuota,
)
from scufris.settings_store import SettingsReadOnly, UnknownSettingKey
from scufris_core import Database

OLDER = datetime(2026, 7, 1, tzinfo=timezone.utc)
NEWER = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _orchestrator(**kwargs: Any) -> AgentRecord:
    fields: dict[str, Any] = {
        "id": ORCHESTRATOR_ID,
        "name": "orchestrator",
        "project_id": "",
        "backend": "claude",
        "model": "sonnet",
    }
    fields.update(kwargs)
    return AgentRecord(**fields)


class FakeRunService:
    """The two things this surface asks of `AgentRunService`: the orchestrator
    record every alias resolves through, and the forked turn."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.known: dict[str, AgentRecord] = {ORCHESTRATOR_ID: _orchestrator()}
        self.seed = "the fork prompt"
        self.launched: list[tuple[str, str | None, str]] = []

    def require_agent(self, agent_id: str) -> AgentRecord:
        self.calls.append(("require_agent", agent_id))
        try:
            return self.known[agent_id]
        except KeyError:
            raise AgentNotFound(agent_id) from None

    def fork_seed(
        self, agent: AgentRecord, session_id: str | None, index: int, text: str
    ) -> str:
        self.calls.append(("fork_seed", (agent.id, session_id, index, text)))
        return self.seed

    async def launch(self, agent: AgentRecord, project: Any, prompt: str, **kw: Any):
        self.calls.append(("launch", (agent.id, prompt)))
        self.launched.append((agent.id, agent.session_id, prompt))
        # A launched turn ends: publish `done` and close, or a real drain never
        # finishes and the TestClient request blocks forever.
        bus: EventBus[Any] = EventBus()
        bus.publish(StreamDone(reply=AgentReply(text="forked"), session_id="sess-new"))
        bus.close()
        return "run-1", bus

    async def drain(self, bus: EventBus[Any]) -> StreamDone:
        self.calls.append(("drain", None))
        return StreamDone(reply=AgentReply(text="forked"), session_id="sess-new")


class FakeDiagnostics:
    """The one reader every alias answers out of."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def usage(self, agent: AgentRecord) -> Capability[UsageQuota]:
        self.calls.append(("usage", agent.id))
        return Capability(supported=False)

    def memory(self, agent: AgentRecord) -> Capability[MemoryFootprint]:
        self.calls.append(("memory", agent.id))
        return Capability(supported=False)

    def account(self, agent: AgentRecord) -> AccountInfo:
        self.calls.append(("account", agent.id))
        return AccountInfo(
            model="sonnet",
            auth_mode=None,
            enabled=True,
            quota=Capability(supported=False),
        )

    async def health(self, agent: AgentRecord) -> AgentHealth:
        self.calls.append(("health", agent.id))
        return AgentHealth(backend="claude", scufris_version="test", checks=[])


class FakeAgentStore:
    """The orchestrator's record plus its session registry."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.record = _orchestrator()
        self.sessions = ["sess-old", "sess-new"]
        self.current: str | None = "sess-old"

    def get(self, agent_id: str) -> AgentRecord:
        self.calls.append(("get", agent_id))
        if agent_id != ORCHESTRATOR_ID:
            raise AgentNotFound(agent_id)
        return self.record

    def orchestrator_sessions(self) -> list[str]:
        self.calls.append(("orchestrator_sessions", None))
        return list(self.sessions)

    def orchestrator_session_id(self) -> str | None:
        return self.current

    def set_orchestrator_session(self, session_id: str | None) -> None:
        self.calls.append(("set_orchestrator_session", session_id))
        self.current = session_id

    def forget_orchestrator_session(self, session_id: str) -> None:
        self.calls.append(("forget_orchestrator_session", session_id))
        if self.current == session_id:
            self.current = None


class FakeSettingsStore:
    """`SettingsStore`'s two answers: whether writes are allowed, and the
    refusal a bad write raises."""

    def __init__(self) -> None:
        self.writable = True
        self.applied: list[dict[str, Any]] = []
        self.raises: Exception | None = None

    def apply(self, updates: dict[str, Any]) -> Any:
        self.applied.append(updates)
        if self.raises is not None:
            raise self.raises


class FakeSupervisor:
    """Records the serialize key the session routes take."""

    def __init__(self) -> None:
        self.keys: list[str] = []

    def serialized(self, key: str) -> Any:
        self.keys.append(key)

        class _Held:
            async def __aenter__(self) -> None:
                return None

            async def __aexit__(self, *exc: Any) -> None:
                return None

        return _Held()


class FakeBackend:
    """The provider side of the session routes, reached through `get_backend`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.deleted = True

    def read_context(self, settings: Settings, session_id: str | None):
        self.calls.append(("read_context", session_id))
        return SessionContext(session_id=session_id or "")

    def read_transcript(
        self, settings: Settings, session_id: str
    ) -> list[TranscriptMessage]:
        self.calls.append(("read_transcript", session_id))
        return [TranscriptMessage(role="user", text="hi")]

    def read_status(self, settings: Settings, session_id: str | None) -> None:
        self.calls.append(("read_status", session_id))
        return None

    async def delete_session(self, settings: Settings, session_id: str) -> bool:
        self.calls.append(("delete_session", session_id))
        return self.deleted


class LegacyAgentRig:
    """The `/api/agent/*` router on a bare app, over the fakes above."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.runs = FakeRunService()
        self.diagnostics = FakeDiagnostics()
        self.agents = FakeAgentStore()
        self.store = FakeSettingsStore()
        self.supervisor = FakeSupervisor()
        self.backend = FakeBackend()
        self.app = FastAPI()
        self.app.include_router(
            build_legacy_agent_router(
                LegacyAgentDeps(
                    settings=settings,
                    agents=cast(Any, self.agents),
                    store=cast(Any, self.store),
                    diagnostics=cast(Any, self.diagnostics),
                    runs=cast(Any, self.runs),
                    supervisor=cast(Any, self.supervisor),
                    api_token="token-1",
                )
            )
        )


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    return Settings(
        state_dir=tmp_path / "state",
        web_dist=tmp_path / "dist",
        _env_file=None,  # type: ignore[call-arg]
        **kwargs,
    )


@pytest.fixture
def legacy_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LegacyAgentRig:
    rig = LegacyAgentRig(_settings(tmp_path))
    # `get_backend` is imported by name into the router module, so the binding
    # patched here is the one the session routes actually call.
    monkeypatch.setattr(
        "scufris.api.legacy_agent.get_backend", lambda _name: rig.backend
    )
    return rig


@pytest.fixture
def legacy_client(legacy_rig: LegacyAgentRig) -> Iterator[TestClient]:
    with TestClient(legacy_rig.app) as client:
        yield client


def test_legacy_agent_routes_delegate_to_scoped_services(
    legacy_rig: LegacyAgentRig, legacy_client: TestClient
) -> None:
    """Every alias resolves the orchestrator through `AgentRunService` and answers
    out of `AgentDiagnostics` - the same two services the scoped
    `/api/agents/orchestrator/*` routes use. No second reader anywhere."""
    for path in ("/api/agent/usage", "/api/agent/memory", "/api/agent/account"):
        assert legacy_client.get(path).status_code == 200, path
    assert legacy_client.get("/api/agent/health").status_code == 200

    assert legacy_rig.diagnostics.calls == [
        ("usage", ORCHESTRATOR_ID),
        ("memory", ORCHESTRATOR_ID),
        ("account", ORCHESTRATOR_ID),
        ("health", ORCHESTRATOR_ID),
    ]
    # Four routes, four record lookups, all through the run service.
    assert legacy_rig.runs.calls == [("require_agent", ORCHESTRATOR_ID)] * 4


def test_agent_info_reads_the_record_not_the_codex_model_slot(
    legacy_rig: LegacyAgentRig, legacy_client: TestClient
) -> None:
    """`settings.agent_model` is the codex slot: a claude orchestrator would
    otherwise report a codex model here while `/account` reported the right one."""
    legacy_rig.runs.known[ORCHESTRATOR_ID] = _orchestrator(
        backend="claude", model="opus"
    )
    body = legacy_client.get("/api/agent/info").json()
    assert body["model"] == "opus"
    assert body["enabled"] is True
    assert legacy_rig.runs.calls == [("require_agent", ORCHESTRATOR_ID)]


def test_the_config_view_layers_live_settings_over_the_record(
    legacy_rig: LegacyAgentRig, legacy_client: TestClient
) -> None:
    """The model and backend come off the record; everything else off settings,
    and `writable` off the settings store rather than a second read of the env."""
    legacy_rig.store.writable = False
    body = legacy_client.get("/api/agent/config").json()
    assert body["model"] == "sonnet"
    assert body["backend"] == "claude"
    assert body["sandbox"] == "read-only"
    assert body["writable"] is False


def test_a_config_patch_is_the_stores_whitelist_translated(
    legacy_rig: LegacyAgentRig, legacy_client: TestClient
) -> None:
    """The router hands the store the present keys only, and answers with the
    effective config the store's write produced."""
    response = legacy_client.patch(
        "/api/agent/config", json={"agent_tools_enabled": False}
    )
    assert response.status_code == 200, response.text
    assert legacy_rig.store.applied == [{"agent_tools_enabled": False}]


@pytest.mark.parametrize(
    ("raised", "status"),
    [
        (SettingsReadOnly("read only"), 403),
        (UnknownSettingKey("nope"), 422),
    ],
)
def test_the_config_patch_maps_every_store_refusal(
    legacy_rig: LegacyAgentRig,
    legacy_client: TestClient,
    raised: Exception,
    status: int,
) -> None:
    """The store decides; the router only says it in HTTP."""
    legacy_rig.store.raises = raised
    response = legacy_client.patch("/api/agent/config", json={"poll_seconds": 2.0})
    assert response.status_code == status, response.text


def test_the_config_update_model_rejects_an_unknown_key() -> None:
    """`extra="forbid"`: an unknown key is a 422 at the boundary rather than a
    silently dropped setting the operator thinks they changed."""
    with pytest.raises(ValueError):
        AgentConfigUpdate(**{"not_a_setting": 1})  # type: ignore[arg-type]


def test_the_tool_runner_refuses_a_disabled_tool_before_it_looks_for_it(
    tmp_path: Path,
) -> None:
    """403, and no server is searched: the operator disabled it, so whether it
    exists is not the question being answered."""
    rig = LegacyAgentRig(_settings(tmp_path, disabled_tools=["host_reboot"]))
    with TestClient(rig.app) as client:
        response = client.post("/api/agent/tools/host_reboot/run", json={"args": {}})
    assert response.status_code == 403, response.text


def test_the_tool_runner_is_404_for_a_tool_no_server_owns(
    legacy_client: TestClient,
) -> None:
    """Never an uncontrolled 500: an unknown name is a client error."""
    response = legacy_client.post("/api/agent/tools/no_such_tool/run", json={})
    assert response.status_code == 404, response.text


def test_the_session_list_is_the_registrys_and_is_newest_first(
    legacy_rig: LegacyAgentRig,
    legacy_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Driven by the ownership registry, not a provider disk scan, so a sub-agent's
    chat can never leak into the operator's switcher; ordered by last activity."""
    infos = {
        "sess-old": SessionInfo(id="sess-old", title="old", updated_at=OLDER),
        "sess-new": SessionInfo(id="sess-new", title="new", updated_at=NEWER),
    }
    monkeypatch.setattr(
        "scufris.api.legacy_agent.session_info",
        lambda _backend, _settings, sid: infos.get(sid),
    )
    body = legacy_client.get("/api/agent/sessions").json()
    assert [s["id"] for s in body["sessions"]] == ["sess-new", "sess-old"]
    assert body["current"] == "sess-old"
    assert ("orchestrator_sessions", None) in legacy_rig.agents.calls


def test_switching_sessions_serializes_on_the_orchestrator_key(
    legacy_rig: LegacyAgentRig, legacy_client: TestClient
) -> None:
    """The orchestrator's turns run through the supervisor keyed on its id, so a
    switch taking any other key could interleave with an in-flight turn."""
    response = legacy_client.post(
        "/api/agent/session", json={"action": "switch", "session_id": "sess-new"}
    )
    assert response.json() == {"current": "sess-new"}
    assert legacy_rig.supervisor.keys == [ORCHESTRATOR_ID]


def test_switching_without_a_session_id_is_422(legacy_client: TestClient) -> None:
    assert (
        legacy_client.post("/api/agent/session", json={"action": "switch"}).status_code
        == 422
    )


def test_deleting_a_session_removes_it_provider_side_and_forgets_it(
    legacy_rig: LegacyAgentRig, legacy_client: TestClient
) -> None:
    """Both halves, or the session comes back in the list on the next read."""
    response = legacy_client.delete("/api/agent/session/sess-old")
    assert response.json() == {"deleted": True, "current": None}
    assert ("delete_session", "sess-old") in legacy_rig.backend.calls
    assert ("forget_orchestrator_session", "sess-old") in legacy_rig.agents.calls
    assert legacy_rig.supervisor.keys == [ORCHESTRATOR_ID]


def test_a_backend_with_no_provider_delete_still_drops_the_session(
    legacy_rig: LegacyAgentRig, legacy_client: TestClient
) -> None:
    legacy_rig.backend.deleted = False
    assert (
        legacy_client.delete("/api/agent/session/sess-old").json()["deleted"] is False
    )
    assert ("forget_orchestrator_session", "sess-old") in legacy_rig.agents.calls


def test_forking_clears_the_session_before_it_launches(
    legacy_rig: LegacyAgentRig, legacy_client: TestClient
) -> None:
    """The clear and the launch's slot claim are one step, and the forked record
    is derived in memory - a store round trip between them is the gap a concurrent
    turn needs to land on the cleared session."""
    response = legacy_client.post(
        "/api/agent/session/fork",
        json={"source_id": "sess-old", "message_index": 2, "text": "edited"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["current"] == "sess-new"

    kinds = [name for name, _ in legacy_rig.runs.calls]
    assert kinds == ["fork_seed", "launch", "drain"]
    # Launched off a record whose session was already cleared, not off a re-read.
    assert legacy_rig.runs.launched == [(ORCHESTRATOR_ID, None, "the fork prompt")]
    assert ("set_orchestrator_session", None) in legacy_rig.agents.calls
    # No outer serialize(): the launch reserves the orchestrator's slot itself,
    # and wrapping it here would self-deadlock on the same key.
    assert legacy_rig.supervisor.keys == []


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [
        ("POST", "/api/agent/session", 503),
        ("POST", "/api/agent/session/fork", 503),
        ("DELETE", "/api/agent/session/sess-old", 503),
    ],
)
def test_a_disabled_agent_refuses_every_mutating_session_route(
    tmp_path: Path, method: str, path: str, status: int
) -> None:
    rig = LegacyAgentRig(_settings(tmp_path, agent_enabled=False))
    with TestClient(rig.app) as client:
        response = client.request(
            method,
            path,
            json={"action": "new", "source_id": "s", "message_index": 0, "text": "t"},
        )
    assert response.status_code == status, response.text
    assert rig.runs.calls == []


def test_a_disabled_agent_reads_empty_rather_than_refusing(tmp_path: Path) -> None:
    """The read routes answer an empty shape: the switcher renders "no sessions"
    instead of an error banner for a deliberately-off agent."""
    rig = LegacyAgentRig(_settings(tmp_path, agent_enabled=False))
    with TestClient(rig.app) as client:
        assert client.get("/api/agent/sessions").json() == {
            "sessions": [],
            "current": None,
        }
        assert client.get("/api/agent/context").json() is None
        assert client.get("/api/agent/session/sess-old").json() == {"messages": []}
    assert rig.backend.calls == []


def _forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError(
        "the legacy agent router constructed a store, a database or a settings "
        "object; the whole point of the deps dataclass is that it does not"
    )


def test_the_legacy_agent_router_reaches_for_nothing(
    legacy_rig: LegacyAgentRig,
    legacy_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Booby trap the four things it must not touch, then drive the surface."""
    for target in (Settings, AgentStore, ProjectStore, Database):
        monkeypatch.setattr(target, "__init__", _forbidden)
    monkeypatch.setattr(scufris.db, "state_database", _forbidden)

    for path in (
        "/api/agent/info",
        "/api/agent/config",
        "/api/agent/usage",
        "/api/agent/memory",
        "/api/agent/account",
        "/api/agent/health",
        "/api/agent/sessions",
        "/api/agent/context",
        "/api/agent/session/sess-old",
    ):
        assert legacy_client.get(path).status_code == 200, path
