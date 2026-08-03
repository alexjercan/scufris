"""The public surface of ``create_app``, pinned.

A CHARACTERIZATION test: it is green on the tree it was written against, by
construction, and its job is to STAY green while the application factory is
taken apart into routers. It asserts nothing about whether the surface is good -
only about what it is - so any drift the extraction introduces (a path renamed,
a response model changed, a route silently dropped off an OpenAPI tag, an
`app.state` key no longer published, a background service no longer started, an
injection seam no longer honoured) shows up as a failure here rather than as a
broken dashboard.

The four things it pins are the four things other code actually depends on:

1. the route table - every path, method, response model, `include_in_schema`
   and assigned tag, plus the two middlewares and their order, and that every
   one of those routes reaches the app through an included ROUTER;
2. the `app.state` keys, read by tests, by the Telegram wiring and by the MCP
   layer;
3. the services the lifespan starts and stops;
4. the five `create_app` override points, asserted by OBSERVING the injected
   object being used rather than by reading the signature.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeCollector, make_fixture_stats
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from scufris.api.routes import iter_api_routes
from scufris.app import create_app
from scufris.config import Settings
from scufris.host import HostOverview
from scufris.hostconfig import ConfigChangeRefused
from scufris.processes import ProcessList

# path, sorted methods, response-model name (None when the endpoint returns a
# Response directly), include_in_schema, OpenAPI tags. Sorted by the tuple, so a
# route added or removed anywhere shows as a one-line diff.
EXPECTED_ROUTES: list[tuple[str, list[str], str | None, bool, list[str]]] = [
    ("/agents/{agent_id}", ["GET"], None, False, []),
    ("/agents/{agent_id}/{rest:path}", ["GET"], None, False, []),
    ("/api/agent/account", ["GET"], "AccountInfo", True, ["sessions"]),
    ("/api/agent/config", ["GET"], "AgentConfig", True, ["settings"]),
    ("/api/agent/config", ["PATCH"], "AgentConfig", True, ["settings"]),
    ("/api/agent/context", ["GET"], "Union", True, ["sessions"]),
    ("/api/agent/health", ["GET"], "AgentHealth", True, ["settings"]),
    ("/api/agent/info", ["GET"], "AgentInfo", True, ["chat"]),
    ("/api/agent/mcp", ["GET"], "list", True, ["settings"]),
    (
        "/api/agent/memory",
        ["GET"],
        "Capability[MemoryFootprint]",
        True,
        ["sessions"],
    ),
    ("/api/agent/session", ["POST"], "CurrentSession", True, ["sessions"]),
    ("/api/agent/session/fork", ["POST"], "ForkResult", True, ["sessions"]),
    ("/api/agent/session/{session_id}", ["DELETE"], "DeleteResult", True, ["sessions"]),
    (
        "/api/agent/session/{session_id}",
        ["GET"],
        "TranscriptResponse",
        True,
        ["sessions"],
    ),
    ("/api/agent/sessions", ["GET"], "SessionsResponse", True, ["sessions"]),
    ("/api/agent/tools", ["GET"], "list", True, ["settings"]),
    ("/api/agent/tools/{name}/run", ["POST"], "ToolRunResult", True, ["settings"]),
    ("/api/agent/usage", ["GET"], "Capability[UsageQuota]", True, ["sessions"]),
    ("/api/agents", ["GET"], "list", True, ["agents"]),
    ("/api/agents", ["POST"], "AgentRecord", True, ["agents"]),
    ("/api/agents/backends", ["GET"], "list", True, ["agents"]),
    ("/api/agents/pending", ["GET"], "list", True, ["agents"]),
    ("/api/agents/{agent_id}", ["DELETE"], "DeleteResult", True, ["agents"]),
    ("/api/agents/{agent_id}", ["GET"], "AgentRecord", True, ["agents"]),
    ("/api/agents/{agent_id}", ["PATCH"], "AgentRecord", True, ["agents"]),
    ("/api/agents/{agent_id}/account", ["GET"], "AccountInfo", True, ["agents"]),
    (
        "/api/agents/{agent_id}/acknowledge",
        ["POST"],
        "AcknowledgeResult",
        True,
        ["agents"],
    ),
    ("/api/agents/{agent_id}/cancel", ["POST"], "CancelResult", True, ["agents"]),
    (
        "/api/agents/{agent_id}/capabilities",
        ["GET"],
        "ProjectCapabilities",
        True,
        ["agents"],
    ),
    ("/api/agents/{agent_id}/chat", ["POST"], None, True, ["agents"]),
    ("/api/agents/{agent_id}/events", ["GET"], None, True, ["agents"]),
    ("/api/agents/{agent_id}/fork", ["POST"], None, True, ["agents"]),
    ("/api/agents/{agent_id}/health", ["GET"], "AgentHealth", True, ["agents"]),
    ("/api/agents/{agent_id}/mcp", ["GET"], "list", True, ["agents"]),
    (
        "/api/agents/{agent_id}/memory",
        ["GET"],
        "Capability[MemoryFootprint]",
        True,
        ["agents"],
    ),
    (
        "/api/agents/{agent_id}/report_back",
        ["POST"],
        "ReportBackResult",
        True,
        ["agents"],
    ),
    (
        "/api/agents/{agent_id}/request_input",
        ["POST"],
        "RequestInputResult",
        True,
        ["agents"],
    ),
    ("/api/agents/{agent_id}/run", ["POST"], "RunStarted", True, ["agents"]),
    ("/api/agents/{agent_id}/status", ["GET"], "AgentRunStatus", True, ["agents"]),
    (
        "/api/agents/{agent_id}/tools",
        ["GET"],
        "Capability[list[AgentTool]]",
        True,
        ["agents"],
    ),
    (
        "/api/agents/{agent_id}/transcript",
        ["GET"],
        "TranscriptResponse",
        True,
        ["agents"],
    ),
    (
        "/api/agents/{agent_id}/usage",
        ["GET"],
        "Capability[UsageQuota]",
        True,
        ["agents"],
    ),
    ("/api/auth/login", ["POST"], None, True, ["auth"]),
    ("/api/auth/logout", ["POST"], None, True, ["auth"]),
    ("/api/auth/session", ["GET"], "AuthSession", True, ["auth"]),
    ("/api/chat", ["POST"], "AgentReply", True, ["chat"]),
    ("/api/chat/reset", ["POST"], "dict", True, ["chat"]),
    ("/api/chat/stream", ["POST"], None, True, ["chat"]),
    ("/api/config", ["GET"], "AppConfig", True, ["app"]),
    ("/api/host/actions", ["GET"], "list", True, ["host"]),
    ("/api/host/actions", ["POST"], "HostActionRecord", True, ["host"]),
    ("/api/host/actions/{action_id}", ["GET"], "HostActionRecord", True, ["host"]),
    (
        "/api/host/actions/{action_id}/approve",
        ["POST"],
        "HostActionLaunched",
        True,
        ["host"],
    ),
    (
        "/api/host/actions/{action_id}/cancel",
        ["POST"],
        "HostActionRecord",
        True,
        ["host"],
    ),
    (
        "/api/host/actions/{action_id}/confirmation",
        ["GET"],
        "Confirmation",
        True,
        ["host"],
    ),
    (
        "/api/host/actions/{action_id}/deny",
        ["POST"],
        "HostActionRecord",
        True,
        ["host"],
    ),
    ("/api/host/actions/{action_id}/events", ["GET"], None, True, ["host"]),
    (
        "/api/host/actions/{action_id}/revert",
        ["POST"],
        "HostActionRecord",
        True,
        ["host"],
    ),
    ("/api/host/audit", ["GET"], "list", True, ["host"]),
    ("/api/host/config/changes", ["GET"], "list", True, ["host"]),
    ("/api/host/config/changes", ["POST"], "ConfigChange", True, ["host"]),
    ("/api/host/config/changes/{change_id}", ["GET"], "ConfigChange", True, ["host"]),
    (
        "/api/host/config/changes/{change_id}/cancel",
        ["POST"],
        "ConfigChange",
        True,
        ["host"],
    ),
    ("/api/host/config/changes/{change_id}/events", ["GET"], None, True, ["host"]),
    ("/api/host/digests", ["GET"], "DigestView", True, ["host"]),
    ("/api/host/digests/run", ["POST"], "DigestView", True, ["host"]),
    ("/api/host/overview", ["GET"], "HostOverview", True, ["host"]),
    ("/api/processes", ["GET"], "ProcessList", True, ["host"]),
    ("/api/projects", ["GET"], "list", True, ["projects"]),
    ("/api/projects", ["POST"], "Project", True, ["projects"]),
    ("/api/projects/discovered", ["GET"], "DiscoveredProjects", True, ["projects"]),
    ("/api/projects/new", ["POST"], "Project", True, ["projects"]),
    ("/api/projects/{project_id}", ["DELETE"], "DeleteResult", True, ["projects"]),
    ("/api/projects/{project_id}", ["GET"], "Project", True, ["projects"]),
    ("/api/projects/{project_id}", ["PATCH"], "Project", True, ["projects"]),
    ("/api/projects/{project_id}/tasks", ["GET"], "list", True, ["projects"]),
    ("/api/stats", ["GET"], "HostStats", True, ["host"]),
    ("/projects/{project_id}", ["GET"], None, False, []),
    ("/projects/{project_id}/{rest:path}", ["GET"], None, False, []),
]

# Registered LAST is outermost: Starlette prepends, so `user_middleware[0]` is
# the first to see a request. The logger must stay outside the auth gate or a
# denial stops being logged.
EXPECTED_MIDDLEWARE = ["log_requests", "enforce_auth"]

# Published before the app ever serves a request. `host_checks_task`,
# `telegram_bot` and `telegram_task` are lifespan-owned and asserted separately.
EXPECTED_STATE_KEYS = [
    "agents",
    "api_token",
    "auth_required",
    "config_changes",
    "config_supervisor",
    "db",
    "digests",
    "host_actions",
    "host_approvals",
    "host_scheduler",
    "host_supervisor",
    "hostd",
    "projects",
    "runs",
    "sessions",
    "supervisor",
    "telegram_approval_ops",
]

LIFESPAN_STATE_KEYS = ["host_checks_task", "telegram_bot", "telegram_task"]


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    """Hermetic settings: no dashboard bundle, no `.env`, state under tmp_path."""
    base: dict[str, Any] = {
        "web_dist": tmp_path / "absent",
        "state_dir": tmp_path,
        "_env_file": None,
    }
    base.update(kwargs)
    return Settings(**base)


def _route_table(app: Any) -> list[tuple[str, list[str], str | None, bool, list[str]]]:
    return sorted(
        (
            route.path,
            sorted(route.methods or ()),
            route.response_model.__name__ if route.response_model is not None else None,
            route.include_in_schema,
            [str(tag) for tag in route.tags],
        )
        for route in iter_api_routes(app)
    )


@pytest.fixture
def app(tmp_path: Path) -> Any:
    return create_app(settings=_settings(tmp_path))


def test_the_public_route_table_is_unchanged(app: Any) -> None:
    """Every path, method, response model, schema visibility and OpenAPI tag.

    The tags are asserted even though `_route_tags` assigns them by PATH after
    every route exists: moving a route onto a router preserves its tag only for
    as long as its path is unchanged, and a router registered with a `prefix=`
    would change the path without anyone noticing at the call site.
    """
    assert _route_table(app) == EXPECTED_ROUTES
    # Both middlewares, in the order that keeps the logger outermost. Part of
    # the same surface: a request reaches a route THROUGH these.
    assert [
        middleware.kwargs["dispatch"].__name__ for middleware in app.user_middleware
    ] == EXPECTED_MIDDLEWARE


def test_application_factory_assembles_domain_routers(app: Any) -> None:
    """`create_app` includes routers; it registers no route of its own.

    The route table above is walked THROUGH the included routers, so it stays
    green whether a route lives on a router or on the app. This is the half it
    cannot see: an `@app.get` left behind (or added back) serves the same path
    while owning its body in the factory, which is the arrangement the split
    exists to end. `app.router.routes` is the app's own list, so an `APIRoute`
    in it was registered on the application object itself - the plain `Route`
    objects FastAPI adds for `/openapi.json`, `/docs` and `/redoc` are not.
    """
    own_routes = [
        route.path for route in app.router.routes if isinstance(route, APIRoute)
    ]
    assert own_routes == []
    assert _route_table(app) == EXPECTED_ROUTES


def test_app_state_publishes_the_keys_other_code_reads(app: Any) -> None:
    """`app.state` is a contract, not a scratchpad.

    Tests, the Telegram wiring and the MCP layer read these by name; a key that
    stops being published fails at runtime with an AttributeError from
    whichever caller happens to run first.
    """
    assert sorted(app.state._state) == EXPECTED_STATE_KEYS


def test_the_lifespan_owns_the_background_services(app: Any) -> None:
    """The scheduler task and the Telegram bot start with serving and stop with it.

    They are started in the lifespan rather than in `create_app` so their loops
    live on the SERVING event loop; the same lifespan cancels them. No bot token
    is configured here, so the bot is published as None - which is itself the
    contract `_deliver_digest` reads.
    """
    with TestClient(app):
        assert sorted(app.state._state) == sorted(
            EXPECTED_STATE_KEYS + LIFESPAN_STATE_KEYS
        )
        checks_task = app.state.host_checks_task
        assert isinstance(checks_task, asyncio.Task)
        assert not checks_task.done()
        assert app.state.telegram_bot is None
        assert app.state.telegram_task is None

    assert checks_task.cancelled()


def test_create_app_honours_its_five_override_points(tmp_path: Path) -> None:
    """The injection seams, asserted by OBSERVING each injected object in use.

    A signature check would pass against a factory that accepted the argument
    and ignored it, which is exactly the regression an extraction can introduce
    (a router built over a freshly constructed default instead of over the
    injected one).
    """
    stats = make_fixture_stats()
    processes = ProcessList(groups=[], total=0)
    overview = HostOverview()

    class FakeProcessCollector:
        def sample(self) -> ProcessList:
            return processes

    class FakeInspector:
        def overview(self) -> HostOverview:
            return overview

    class FakeBuilder:
        """Only the seam matters here: that THIS object is the one asked to
        resolve a ref. Refusing is the cheapest observable answer."""

        resolved: list[str] = []

        def resolve(self, repo: Path, ref: str, *, allowed: Path) -> Any:
            FakeBuilder.resolved.append(ref)
            raise ConfigChangeRefused("the injected builder was asked")

    settings = _settings(tmp_path, poll_seconds=7.25)
    app = create_app(
        collector=FakeCollector(stats),
        settings=settings,
        process_collector=FakeProcessCollector(),  # type: ignore[arg-type]
        config_builder=FakeBuilder(),  # type: ignore[arg-type]
        host_inspector=FakeInspector(),  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        assert client.get("/api/stats").json()["hostname"] == stats.hostname
        assert client.get("/api/processes").json() == processes.model_dump(mode="json")
        assert client.get("/api/host/overview").json() == overview.model_dump(
            mode="json"
        )
        # settings: the client-facing config is read off the injected object.
        assert client.get("/api/config").json()["poll_seconds"] == 7.25
        # config_builder: reached, and its refusal is what the route reports.
        refused = client.post("/api/host/config/changes", json={"ref": "abc123"})
        assert refused.status_code == 422, refused.text
    assert FakeBuilder.resolved == ["abc123"]
