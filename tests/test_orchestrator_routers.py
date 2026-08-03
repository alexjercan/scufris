"""The orchestrator routers, driven over fakes on a bare app.

The companion to `test_domain_routers.py`, which does the same for the auth,
host and host-config routers. It lives in its own file rather than extending
that one because that file is at its line cap; the rig pattern is the same.

Two claims, and they are the two halves of one:

- each router TRANSLATES - it calls the store or the service that owns a rule and
  turns its refusal into a status, so a rule reappearing in a route body shows up
  here as a call that never happened;
- `test_orchestrator_routers_reach_for_nothing` - a router constructs no
  `Settings`, no store and no `Database`. The four are booby trapped for the
  duration and the whole surface is driven through them.

The fakes are deliberately shallow: what is worth proving about a route is which
collaborator it asks and which status it maps the answer to, not that the
collaborator is right - that is the collaborator's own test.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import scufris.db
from scufris.agent import AgentReply, StreamDone
from scufris.agent_diagnostics import AccountInfo
from scufris.agent_store import (
    ORCHESTRATOR_ID,
    AgentNotFound,
    AgentRecord,
    AgentsReadOnly,
    AgentStore,
    InvalidAgent,
    ReservedAgent,
)
from scufris.api.agent_runs import AgentRunDeps, build_agent_run_router
from scufris.api.agents import AgentDeps, build_agent_router
from scufris.api.projects import ProjectDeps, build_project_router
from scufris.backends import Capability
from scufris.config import Settings
from scufris.enums import AgentState
from scufris.eventbus import EventBus
from scufris.orchestrator import AgentDisabled, RunAlreadyActive
from scufris.orchestrator.runs import TurnStatus
from scufris.projects import (
    DuplicateProject,
    InvalidProject,
    Project,
    ProjectNotFound,
    ProjectsReadOnly,
    ProjectStore,
)
from scufris.sessions import MemoryFootprint, UsageQuota
from scufris.settings_store import SettingsReadOnly
from scufris_core import Database

PROJECT_ID = "proj-1"


def _project(project_id: str = PROJECT_ID, *, cwd: str = "/srv/thing") -> Project:
    return Project(id=project_id, name="thing", cwd=cwd, language="python")


class FakeProjects:
    """The project store's answers, recorded. Every refusal it can raise is
    scripted by name, because the statuses are the whole contract."""

    def __init__(self) -> None:
        self.writable = True
        self.rows = [_project()]
        self.calls: list[tuple[str, Any]] = []
        self.raises: Exception | None = None

    def _maybe_raise(self) -> None:
        if self.raises is not None:
            raise self.raises

    def list(self) -> list[Project]:
        self.calls.append(("list", None))
        return list(self.rows)

    def get(self, project_id: str) -> Project:
        self.calls.append(("get", project_id))
        self._maybe_raise()
        return _project(project_id)

    def create(self, **kwargs: Any) -> Project:
        self.calls.append(("create", kwargs))
        self._maybe_raise()
        return _project()

    def update(self, project_id: str, **kwargs: Any) -> Project:
        self.calls.append(("update", (project_id, kwargs)))
        self._maybe_raise()
        return _project(project_id)

    def delete(self, project_id: str) -> None:
        self.calls.append(("delete", project_id))
        self._maybe_raise()


class ProjectRig:
    """The project router on a bare app, with the store fake reachable by name."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.projects = FakeProjects()
        self.app = FastAPI()
        self.app.include_router(
            build_project_router(
                ProjectDeps(settings=settings, projects=cast(Any, self.projects))
            )
        )


@pytest.fixture
def project_rig(tmp_path: Path) -> ProjectRig:
    # Built HERE and handed to the router. The router never constructs one -
    # `test_orchestrator_routers_reach_for_nothing` is what holds it to that.
    return ProjectRig(
        Settings(
            state_dir=tmp_path / "state",
            web_dist=tmp_path / "dist",
            project_base_dirs=[tmp_path / "code"],
            _env_file=None,  # type: ignore[call-arg]
        )
    )


@pytest.fixture
def project_client(project_rig: ProjectRig) -> Iterator[TestClient]:
    with TestClient(project_rig.app) as client:
        yield client


def test_project_routes_delegate_to_the_store(
    project_rig: ProjectRig, project_client: TestClient
) -> None:
    """Every project route asks the store and reports what it answered."""
    assert project_client.get("/api/projects").json() == [
        _project().model_dump(mode="json")
    ]
    assert project_client.get(f"/api/projects/{PROJECT_ID}").json()["id"] == PROJECT_ID
    created = project_client.post(
        "/api/projects", json={"name": "thing", "cwd": "/srv/thing"}
    )
    assert created.status_code == 200, created.text
    patched = project_client.patch(
        f"/api/projects/{PROJECT_ID}", json={"name": "renamed"}
    )
    assert patched.status_code == 200, patched.text
    deleted = project_client.request("DELETE", f"/api/projects/{PROJECT_ID}")
    assert deleted.json() == {"deleted": True, "current": None}

    assert [name for name, _ in project_rig.projects.calls] == [
        "list",
        "get",
        "create",
        "update",
        "delete",
    ]


@pytest.mark.parametrize(
    ("raised", "status"),
    [
        (ProjectsReadOnly("read only"), 403),
        (InvalidProject("bad cwd"), 422),
        (DuplicateProject("already"), 409),
    ],
)
def test_project_creation_maps_every_store_refusal(
    project_rig: ProjectRig,
    project_client: TestClient,
    raised: Exception,
    status: int,
) -> None:
    """The store owns the rule; the route owns only the status it becomes."""
    project_rig.projects.raises = raised
    response = project_client.post(
        "/api/projects", json={"name": "thing", "cwd": "/srv/thing"}
    )
    assert response.status_code == status, response.text


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", f"/api/projects/{PROJECT_ID}"),
        ("PATCH", f"/api/projects/{PROJECT_ID}"),
        ("DELETE", f"/api/projects/{PROJECT_ID}"),
        ("GET", f"/api/projects/{PROJECT_ID}/tasks"),
    ],
)
def test_an_unknown_project_is_a_404_on_every_route_that_names_one(
    project_rig: ProjectRig, project_client: TestClient, method: str, path: str
) -> None:
    """`ProjectNotFound` is the store's answer; 404 is this router's word for it."""
    project_rig.projects.raises = ProjectNotFound("nope")
    response = project_client.request(method, path, json={})
    assert response.status_code == 404, response.text


def test_the_discovery_view_unions_discovered_dirs_with_registered_projects(
    project_rig: ProjectRig, project_client: TestClient, tmp_path: Path
) -> None:
    """The Projects page's source of truth: what is on disk under a base dir,
    flagged with whether it is registered, PLUS registered projects that live
    outside every base dir - which would otherwise be invisible on the page."""
    base = tmp_path / "code"
    (base / "found").mkdir(parents=True)
    project_rig.projects.rows = [_project(cwd=str(base / "found")), _project("p2")]

    body = project_client.get("/api/projects/discovered").json()
    by_name = {row["path"]: row for row in body["projects"]}
    assert by_name[str(base / "found")]["registered"] is True
    # Registered outside a base dir: added rather than dropped.
    assert by_name[str(Path("/srv/thing"))]["project_id"] == "p2"
    assert body["base_dirs"] == [str(base)]


def test_creating_a_new_project_refuses_a_base_outside_the_configured_dirs(
    project_client: TestClient, tmp_path: Path
) -> None:
    """422 BEFORE any mkdir. The base allowlist is the only thing standing
    between this endpoint and creating a directory anywhere on the box."""
    outside = tmp_path / "elsewhere"
    response = project_client.post(
        "/api/projects/new", json={"name": "thing", "base": str(outside)}
    )
    assert response.status_code == 422, response.text
    assert not outside.exists()


def test_creating_a_new_project_refuses_a_read_only_store_before_the_mkdir(
    project_rig: ProjectRig, project_client: TestClient, tmp_path: Path
) -> None:
    """A refused request must not leave a directory behind as a side effect."""
    project_rig.projects.writable = False
    response = project_client.post(
        "/api/projects/new", json={"name": "thing", "base": str(tmp_path / "code")}
    )
    assert response.status_code == 403, response.text
    assert not (tmp_path / "code" / "thing").exists()


def test_project_tasks_are_read_off_the_projects_working_tree(
    project_client: TestClient,
) -> None:
    """The route resolves the project first and reads ITS cwd; a tree with no
    `tasks/` yields an empty list rather than walking up to a parent's."""
    assert project_client.get(f"/api/projects/{PROJECT_ID}/tasks").json() == []


def test_the_project_detail_shell_is_404_until_the_frontend_is_built(
    project_client: TestClient,
) -> None:
    """A page, not an API: it 404s honestly rather than serving the API index."""
    assert project_client.get(f"/projects/{PROJECT_ID}").status_code == 404
    assert project_client.get(f"/projects/{PROJECT_ID}/settings").status_code == 404


def _forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError(
        "a router constructed a store, a database or a settings object; the whole "
        "point of the deps dataclass is that it does not"
    )


def test_orchestrator_routers_reach_for_nothing(
    project_rig: ProjectRig,
    project_client: TestClient,
    agent_rig: AgentRecordRig,
    agent_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Booby trap the four things a router must not touch, then drive the surface.

    The traps go on ``__init__`` rather than on the name in the defining module:
    `from ..config import Settings` binds the class into the importing module at
    import time, so replacing `scufris.config.Settings` would leave the router
    calling the real one and the trap would never fire. Patching the class itself
    catches every import spelling.
    """
    for target in (Settings, AgentStore, ProjectStore, Database):
        monkeypatch.setattr(target, "__init__", _forbidden)
    monkeypatch.setattr(scufris.db, "state_database", _forbidden)

    for path in (
        "/api/projects",
        "/api/projects/discovered",
        f"/api/projects/{PROJECT_ID}",
        f"/api/projects/{PROJECT_ID}/tasks",
    ):
        assert project_client.get(path).status_code == 200, path
    assert (
        project_client.post(
            "/api/projects", json={"name": "thing", "cwd": "/srv/thing"}
        ).status_code
        == 200
    )

    for path in (
        "/api/agents",
        "/api/agents/backends",
        "/api/agents/pending",
        f"/api/agents/{AGENT_ID}",
        f"/api/agents/{AGENT_ID}/capabilities",
    ):
        assert agent_client.get(path).status_code == 200, path
    assert (
        agent_client.post(
            "/api/agents", json={"name": "worker", "project_id": PROJECT_ID}
        ).status_code
        == 200
    )


# --- running an agent -------------------------------------------------------

AGENT_ID = "agent-1"


def _agent(agent_id: str = AGENT_ID, **kwargs: Any) -> AgentRecord:
    fields: dict[str, Any] = {
        "id": agent_id,
        "name": "worker",
        "project_id": PROJECT_ID,
        "backend": "claude",
        "goal": "ship it",
    }
    fields.update(kwargs)
    return AgentRecord(**fields)


class FakeRunService:
    """`AgentRunService`'s answers, recorded. Every refusal is scripted by name
    because the statuses are what the router is responsible for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.known = {AGENT_ID: _agent()}
        self.bus: EventBus[Any] = EventBus()
        self.launch_raises: Exception | None = None
        self.acknowledged = True
        self.project: Project | None = _project()

    def require_agent(self, agent_id: str) -> AgentRecord:
        self.calls.append(("require_agent", agent_id))
        try:
            return self.known[agent_id]
        except KeyError:
            raise AgentNotFound(agent_id) from None

    def require_agent_project(self, agent: AgentRecord) -> Project | None:
        self.calls.append(("require_agent_project", agent.id))
        return self.project

    async def launch(self, agent: AgentRecord, project: Any, prompt: str, **kw: Any):
        self.calls.append(("launch", (agent.id, prompt)))
        if self.launch_raises is not None:
            raise self.launch_raises
        # A launched turn ends: publish `done` and close, or the chat route's SSE
        # relay never finishes and the TestClient request blocks forever.
        bus: EventBus[Any] = EventBus()
        bus.publish(StreamDone(reply=AgentReply(text="ok")))
        bus.close()
        self.bus = bus
        return "run-1", bus

    def cancel(self, agent_id: str) -> None:
        self.calls.append(("cancel", agent_id))

    def status(self, agent: AgentRecord) -> TurnStatus:
        self.calls.append(("status", agent.id))
        return TurnStatus(
            agent_id=agent.id,
            state="running",
            session_id="sess-1",
            progress=None,
            prompt="ship it",
        )

    def request_input(self, agent: AgentRecord, question: str) -> AgentState:
        self.calls.append(("request_input", (agent.id, question)))
        return AgentState.WAITING

    def report_back(self, agent: AgentRecord, summary: str) -> AgentState:
        self.calls.append(("report_back", (agent.id, summary)))
        return AgentState.REPORTED

    async def acknowledge(self, agent_id: str) -> bool:
        self.calls.append(("acknowledge", agent_id))
        return self.acknowledged


class FakeApprovalsForAgents:
    """Only the one question the run router asks of the approval service."""

    def __init__(self) -> None:
        self.live: Any = None

    async def live_for_agent(self, agent_id: str) -> Any:
        return self.live


class FakeAgentGate:
    def __init__(self) -> None:
        self.is_agent = False

    async def caller_is_agent(self, request: Any) -> bool:
        return self.is_agent


class FakeSupervisor:
    def status(self, run_id: str) -> Any:
        return SimpleNamespace(state="queued")


class FakeDiagnostics:
    def usage(self, agent: AgentRecord) -> Capability[UsageQuota]:
        return Capability(supported=False)

    def memory(self, agent: AgentRecord) -> Capability[MemoryFootprint]:
        return Capability(supported=False)

    def account(self, agent: AgentRecord) -> AccountInfo:
        return AccountInfo(
            model="m",
            auth_mode=None,
            enabled=True,
            quota=Capability(supported=False),
        )


class FakeAgentStore:
    def __init__(self) -> None:
        self.spawn_parents: list[tuple[str, str, str]] = []

    def record_spawn_parent(self, agent_id: str, parent: str, session: str) -> None:
        self.spawn_parents.append((agent_id, parent, session))


class AgentRunRig:
    """The agent-run router on a bare app, over the fakes above."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.runs = FakeRunService()
        self.approvals = FakeApprovalsForAgents()
        self.gate = FakeAgentGate()
        self.agents = FakeAgentStore()
        self.app = FastAPI()
        self.app.include_router(
            build_agent_run_router(
                AgentRunDeps(
                    settings=settings,
                    agents=cast(Any, self.agents),
                    runs=cast(Any, self.runs),
                    diagnostics=cast(Any, FakeDiagnostics()),
                    gate=cast(Any, self.gate),
                    approvals=cast(Any, self.approvals),
                    supervisor=cast(Any, FakeSupervisor()),
                )
            )
        )


@pytest.fixture
def run_rig(tmp_path: Path) -> AgentRunRig:
    return AgentRunRig(
        Settings(
            state_dir=tmp_path / "state",
            web_dist=tmp_path / "dist",
            _env_file=None,  # type: ignore[call-arg]
        )
    )


@pytest.fixture
def run_client(run_rig: AgentRunRig) -> Iterator[TestClient]:
    with TestClient(run_rig.app) as client:
        yield client


def test_launching_a_run_reports_the_supervisors_own_state(
    run_rig: AgentRunRig, run_client: TestClient
) -> None:
    """ "queued" until a slot is free, not an assumed "running" - the difference
    is what the agents page shows while the pool is saturated."""
    body = run_client.post(f"/api/agents/{AGENT_ID}/run", json={}).json()
    assert body == {"agent_id": AGENT_ID, "state": "queued"}
    assert ("launch", (AGENT_ID, "ship it")) in run_rig.runs.calls


def test_a_run_with_no_goal_anywhere_is_refused_before_it_launches(
    run_rig: AgentRunRig, run_client: TestClient
) -> None:
    run_rig.runs.known = {AGENT_ID: _agent(goal="")}
    response = run_client.post(f"/api/agents/{AGENT_ID}/run", json={})
    assert response.status_code == 422, response.text
    assert not [name for name, _ in run_rig.runs.calls if name == "launch"]


def test_a_parent_session_stamps_the_child_before_the_launch(
    run_rig: AgentRunRig, run_client: TestClient
) -> None:
    """Part 3: a later request_input has to route back to the chat that spawned
    the child, and the stamp is the only record of which chat that was."""
    run_client.post(f"/api/agents/{AGENT_ID}/run", json={"parent_session_id": "chat-7"})
    assert run_rig.agents.spawn_parents == [(AGENT_ID, ORCHESTRATOR_ID, "chat-7")]


def test_the_run_router_maps_every_orchestrator_refusal(
    run_rig: AgentRunRig, run_client: TestClient
) -> None:
    """The service decides; the router only says it in HTTP."""
    run_rig.runs.launch_raises = RunAlreadyActive("already running")
    assert run_client.post(f"/api/agents/{AGENT_ID}/run", json={}).status_code == 409

    run_rig.runs.launch_raises = AgentDisabled("agent is disabled")
    assert run_client.post(f"/api/agents/{AGENT_ID}/run", json={}).status_code == 503


def test_an_unknown_agent_is_a_404_across_the_run_surface(
    run_client: TestClient,
) -> None:
    for method, path, body in (
        ("POST", "/api/agents/ghost/run", {}),
        ("POST", "/api/agents/ghost/cancel", None),
        ("GET", "/api/agents/ghost/status", None),
        ("POST", "/api/agents/ghost/request_input", {"question": "?"}),
        ("POST", "/api/agents/ghost/report_back", {"summary": "done"}),
        ("GET", "/api/agents/ghost/usage", None),
        ("GET", "/api/agents/ghost/account", None),
    ):
        response = run_client.request(method, path, json=body)
        assert response.status_code == 404, path


def test_the_signal_routes_refuse_an_empty_body(run_client: TestClient) -> None:
    """An empty question or summary is the signal saying nothing, which is worse
    than not signalling: the orchestrator would poll a row it cannot act on."""
    assert (
        run_client.post(
            f"/api/agents/{AGENT_ID}/request_input", json={"question": "  "}
        ).status_code
        == 422
    )
    assert (
        run_client.post(
            f"/api/agents/{AGENT_ID}/report_back", json={"summary": " "}
        ).status_code
        == 422
    )


def test_a_live_host_approval_refuses_the_orchestrator_but_not_the_operator(
    run_rig: AgentRunRig, run_client: TestClient
) -> None:
    """The agent is waiting for the OPERATOR. A resume from the orchestrator
    carrying "approved, go ahead" is an answer it has no authority to give -
    but the operator reading the agent's own chat is not deciding anything.
    """
    run_rig.approvals.live = SimpleNamespace(id="act-1")

    run_rig.gate.is_agent = True
    refused = run_client.post(f"/api/agents/{AGENT_ID}/chat", json={"message": "go"})
    assert refused.status_code == 409, refused.text
    assert "act-1" in refused.json()["detail"]

    run_rig.gate.is_agent = False
    assert run_client.post(
        f"/api/agents/{AGENT_ID}/chat", json={"message": "go"}
    ).status_code == (200)


def test_a_live_host_approval_cannot_be_acknowledged_away(
    run_rig: AgentRunRig, run_client: TestClient
) -> None:
    """Acknowledging would hide a decision nobody has made. Once the proposal is
    decided or expired the signal acknowledges like any other."""
    run_rig.approvals.live = SimpleNamespace(id="act-1")
    assert run_client.post(f"/api/agents/{AGENT_ID}/acknowledge").json() == {
        "agent_id": AGENT_ID,
        "acknowledged": False,
    }
    assert not [n for n, _ in run_rig.runs.calls if n == "acknowledge"]

    run_rig.approvals.live = None
    assert run_client.post(f"/api/agents/{AGENT_ID}/acknowledge").json() == {
        "agent_id": AGENT_ID,
        "acknowledged": True,
    }


def test_the_orchestrator_cannot_revert_fork_through_the_agent_route(
    run_client: TestClient,
) -> None:
    """It keeps its multi-session fork; a revert-fork would drop its history."""
    response = run_client.post(
        f"/api/agents/{ORCHESTRATOR_ID}/fork", json={"message_index": 0, "text": "x"}
    )
    assert response.status_code in (404, 409), response.text


# --- the agent record surface ------------------------------------------------


class FakeAgentRecords:
    """The agent store's answers, recorded. Every refusal scripted by name."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.rows = [_agent()]
        self.raises: Exception | None = None
        self.pending: dict[str, Any] = {}
        self.parents: dict[str, tuple[str | None, str | None]] = {}

    def _maybe_raise(self) -> None:
        if self.raises is not None:
            raise self.raises

    def list(self) -> list[AgentRecord]:
        self.calls.append(("list", None))
        return list(self.rows)

    def get(self, agent_id: str) -> AgentRecord:
        self.calls.append(("get", agent_id))
        self._maybe_raise()
        return _agent(agent_id)

    def create(self, **kwargs: Any) -> AgentRecord:
        self.calls.append(("create", kwargs))
        self._maybe_raise()
        return _agent()

    def update(self, agent_id: str, **kwargs: Any) -> AgentRecord:
        self.calls.append(("update", (agent_id, kwargs)))
        self._maybe_raise()
        return _agent(agent_id)

    def delete(self, agent_id: str) -> None:
        self.calls.append(("delete", agent_id))
        self._maybe_raise()

    def pending_outcomes(self) -> dict[str, Any]:
        self.calls.append(("pending_outcomes", None))
        return dict(self.pending)

    def parent_of(self, agent_id: str) -> tuple[str | None, str | None]:
        return self.parents.get(agent_id, (None, None))


class FakeSettingsStore:
    """The settings store, for the ONE write the record router makes: the
    orchestrator's own config, which has no agents row to update."""

    def __init__(self) -> None:
        self.applied: list[dict[str, Any]] = []
        self.raises: Exception | None = None

    def apply(self, updates: dict[str, Any]) -> None:
        if self.raises is not None:
            raise self.raises
        self.applied.append(updates)


def _outcome(state: AgentState = AgentState.WAITING, ts: float = 1.0) -> Any:
    return SimpleNamespace(
        state=state, message="?", run_id="run-1", session_id="sess-1", ts=ts
    )


class AgentRecordRig:
    """The agent record router on a bare app, over the fakes above."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.agents = FakeAgentRecords()
        self.store = FakeSettingsStore()
        self.runs = FakeRunService()
        self.app = FastAPI()
        self.app.include_router(
            build_agent_router(
                AgentDeps(
                    settings=settings,
                    agents=cast(Any, self.agents),
                    store=cast(Any, self.store),
                    runs=cast(Any, self.runs),
                )
            )
        )


@pytest.fixture
def agent_rig(tmp_path: Path) -> AgentRecordRig:
    return AgentRecordRig(
        Settings(
            state_dir=tmp_path / "state",
            web_dist=tmp_path / "dist",
            _env_file=None,  # type: ignore[call-arg]
        )
    )


@pytest.fixture
def agent_client(agent_rig: AgentRecordRig) -> Iterator[TestClient]:
    with TestClient(agent_rig.app) as client:
        yield client


def test_agent_record_routes_delegate_to_the_store(
    agent_rig: AgentRecordRig, agent_client: TestClient
) -> None:
    """Every record route asks the store and reports what it answered."""
    assert agent_client.get("/api/agents").json() == [_agent().model_dump(mode="json")]
    assert agent_client.get(f"/api/agents/{AGENT_ID}").json()["id"] == AGENT_ID
    created = agent_client.post(
        "/api/agents", json={"name": "worker", "project_id": PROJECT_ID}
    )
    assert created.status_code == 200, created.text
    patched = agent_client.patch(f"/api/agents/{AGENT_ID}", json={"name": "renamed"})
    assert patched.status_code == 200, patched.text
    deleted = agent_client.request("DELETE", f"/api/agents/{AGENT_ID}")
    assert deleted.json() == {"deleted": True, "current": None}

    assert [name for name, _ in agent_rig.agents.calls] == [
        "list",
        "get",
        "create",
        "update",
        "delete",
    ]


@pytest.mark.parametrize(
    ("raised", "status"),
    [
        (AgentsReadOnly("read only"), 403),
        (InvalidAgent("no such project"), 422),
    ],
)
def test_agent_creation_maps_every_store_refusal(
    agent_rig: AgentRecordRig,
    agent_client: TestClient,
    raised: Exception,
    status: int,
) -> None:
    """The store owns the rule; the route owns only the status it becomes."""
    agent_rig.agents.raises = raised
    response = agent_client.post(
        "/api/agents", json={"name": "worker", "project_id": PROJECT_ID}
    )
    assert response.status_code == status, response.text


@pytest.mark.parametrize(
    ("raised", "status"),
    [
        (AgentsReadOnly("read only"), 403),
        (ReservedAgent("reserved"), 409),
        (AgentNotFound("ghost"), 404),
        (InvalidAgent("bad backend"), 422),
    ],
)
def test_agent_update_maps_every_store_refusal(
    agent_rig: AgentRecordRig,
    agent_client: TestClient,
    raised: Exception,
    status: int,
) -> None:
    agent_rig.agents.raises = raised
    response = agent_client.patch(f"/api/agents/{AGENT_ID}", json={"name": "x"})
    assert response.status_code == status, response.text


def test_deleting_a_reserved_agent_is_refused_rather_than_conflicting(
    agent_rig: AgentRecordRig, agent_client: TestClient
) -> None:
    """A reserved agent cannot be deleted at all, so the answer is 403 - the
    delete surface has no "resolve the conflict and retry"."""
    agent_rig.agents.raises = ReservedAgent("orchestrator is reserved")
    response = agent_client.request("DELETE", f"/api/agents/{ORCHESTRATOR_ID}")
    assert response.status_code == 403, response.text


def test_an_unknown_agent_is_a_404_across_the_record_surface(
    agent_rig: AgentRecordRig, agent_client: TestClient
) -> None:
    agent_rig.agents.raises = AgentNotFound("ghost")
    for method, path, body in (
        ("GET", "/api/agents/ghost", None),
        ("PATCH", "/api/agents/ghost", {"name": "x"}),
        ("DELETE", "/api/agents/ghost", None),
    ):
        response = agent_client.request(method, path, json=body)
        assert response.status_code == 404, path
    assert agent_client.get("/api/agents/ghost/capabilities").status_code == 404


def test_the_backend_picker_is_server_authoritative(agent_client: TestClient) -> None:
    """The create/settings pickers offer what the SERVER allows, each row
    carrying the default model stamped when it is chosen."""
    rows = agent_client.get("/api/agents/backends").json()
    assert rows, "no backend is selectable at all"
    assert {"id", "label", "default_model", "models"} <= set(rows[0])
    assert "mock" not in {row["id"] for row in rows}


def test_backends_and_pending_are_not_parsed_as_agent_ids(
    agent_rig: AgentRecordRig, agent_client: TestClient
) -> None:
    """Both are declared ahead of `/api/agents/{id}`; if that ordering is lost
    they become a store lookup for an agent named "backends"."""
    assert agent_client.get("/api/agents/backends").status_code == 200
    assert agent_client.get("/api/agents/pending").status_code == 200
    assert not [n for n, _ in agent_rig.agents.calls if n == "get"]


def test_pending_agents_are_scoped_to_the_asking_chat(
    agent_rig: AgentRecordRig, agent_client: TestClient
) -> None:
    """A poll from chat A sees its own children and unattributed ones, never
    chat B's - and the rows come back newest first."""
    agent_rig.agents.pending = {
        "mine": _outcome(ts=1.0),
        "theirs": _outcome(ts=2.0),
        "orphan": _outcome(ts=3.0),
    }
    agent_rig.agents.parents = {
        "mine": (ORCHESTRATOR_ID, "chat-a"),
        "theirs": (ORCHESTRATOR_ID, "chat-b"),
        "orphan": (None, None),
    }

    rows = agent_client.get("/api/agents/pending?parent_session_id=chat-a").json()
    assert [row["agent_id"] for row in rows] == ["orphan", "mine"]
    assert rows[1]["parent_session_id"] == "chat-a"

    # No filter keeps everything (back-compat with a poll that names no chat).
    unfiltered = agent_client.get("/api/agents/pending").json()
    assert {row["agent_id"] for row in unfiltered} == {"mine", "theirs", "orphan"}


def test_the_orchestrators_edits_route_to_the_settings_store(
    agent_rig: AgentRecordRig, agent_client: TestClient
) -> None:
    """It has no agents row: its config lives in the settings store, and the
    model key follows the EFFECTIVE backend rather than always `agent_model`."""
    agent_rig.runs.known[ORCHESTRATOR_ID] = _agent(ORCHESTRATOR_ID)
    response = agent_client.patch(
        f"/api/agents/{ORCHESTRATOR_ID}",
        json={"backend": "claude", "model": "sonnet"},
    )
    assert response.status_code == 200, response.text
    assert agent_rig.store.applied == [
        {"agent_backend": "claude", "claude_model": "sonnet"}
    ]
    # Read back through the synthetic record, not through `agents.update`.
    assert not [n for n, _ in agent_rig.agents.calls if n == "update"]
    assert ("get", ORCHESTRATOR_ID) in agent_rig.agents.calls


def test_a_read_only_settings_store_refuses_the_orchestrators_edit(
    agent_rig: AgentRecordRig, agent_client: TestClient
) -> None:
    agent_rig.store.raises = SettingsReadOnly("read only")
    response = agent_client.patch(
        f"/api/agents/{ORCHESTRATOR_ID}", json={"model": "sonnet"}
    )
    assert response.status_code == 403, response.text


def test_a_project_less_agent_has_an_empty_capability_set(
    agent_rig: AgentRecordRig, agent_client: TestClient
) -> None:
    """No project tree to read skills and custom tools out of - an empty set
    rather than a 404 or the host's own recipes."""
    agent_rig.runs.project = None
    body = agent_client.get(f"/api/agents/{AGENT_ID}/capabilities").json()
    assert body == {"skills": [], "tools": []}


def test_the_agent_detail_shell_is_404_until_the_frontend_is_built(
    agent_client: TestClient,
) -> None:
    """A page, not an API: it 404s honestly rather than serving the API index."""
    assert agent_client.get(f"/agents/{AGENT_ID}").status_code == 404
    assert agent_client.get(f"/agents/{AGENT_ID}/settings").status_code == 404
