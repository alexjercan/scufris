"""How the API documents itself: the description, the tag map, and the tagging.

Routes are grouped in /docs and /redoc by OpenAPI tag, and the tag is assigned by
PATH from the single map in `route_tags` rather than by a `tags=` on every
decorator - so the grouping is one thing to read and one thing to change, and a
route that moves between routers keeps its tag as long as its path is unchanged.

`apply_route_tags` is the pass that does the assigning, and it must run AFTER
every router is included: it walks the app's real surface through `iter_routes`,
which resolves included routers, rather than `app.routes`, where an included
router is one opaque node and every routed endpoint would go silently untagged.
"""

from __future__ import annotations

from typing import Any

from .routes import iter_api_routes

# Shown at the top of /docs (Swagger) and /redoc. Markdown is rendered there.
API_DESCRIPTION = """\
The Scufris backend: a host dashboard and a multi-agent orchestrator.

It serves live host metrics, the main **orchestrator agent** chat (streamed over
SSE), first-class **projects**, and the **agents** that run on them - each agent
is bound to a project, driven by a swappable backend (codex or Claude Code), and
run as a supervised background job with live status and an event stream.

Endpoints are grouped by the tags below. Mutating endpoints under a writable
server are gated by `SCUFRIS_SETTINGS_WRITABLE`; agent turns run read-only unless
an agent has the per-agent write opt-in enabled.
"""

# Tag metadata drives the section ORDER and descriptions in /docs. Routes are
# assigned to these tags by path in `route_tags` (below), so a single map keeps
# the grouping in one place rather than a `tags=` on every decorator.
OPENAPI_TAGS: list[dict[str, str]] = [
    {
        "name": "auth",
        "description": "The operator session: log in, log out, and ask whether authentication is required at all.",
    },
    {
        "name": "host",
        "description": "Read-only host inspection: the live metrics snapshot (stats, processes) and the deeper overview - failed units, NixOS generations, storage and thermals.",
    },
    {
        "name": "app",
        "description": "Client-facing app configuration (poll interval, agent on/off).",
    },
    {
        "name": "chat",
        "description": "The main orchestrator agent chat - one turn (`/api/chat`) or streamed live over SSE (`/api/chat/stream`).",
    },
    {
        "name": "sessions",
        "description": "The chat agent's codex sessions: list, switch, fork, transcript, context window, usage/quota, on-disk memory and account.",
    },
    {
        "name": "settings",
        "description": "Agent configuration: effective config, the tool catalog and health checks.",
    },
    {
        "name": "projects",
        "description": "First-class projects (a workspace an agent runs in) and their tatr tasks.",
    },
    {
        "name": "agents",
        "description": "The multi-agent orchestrator: agent records (CRUD) and running them - launch a goal, poll status, stream events.",
    },
]


def route_tags(path: str) -> list[str]:
    """The OpenAPI tag for an API route, by path (see OPENAPI_TAGS).

    Order matters: the session/context family and the singular `/api/agent/...`
    settings family share a prefix, and the plural `/api/agents` must not be
    caught by the singular check.
    """
    if path.startswith("/api/auth/"):
        return ["auth"]
    if path in ("/api/stats", "/api/processes") or path.startswith("/api/host/"):
        return ["host"]
    if path == "/api/config":
        return ["app"]
    if path.startswith("/api/chat") or path == "/api/agent/info":
        return ["chat"]
    if path.startswith("/api/agents"):
        return ["agents"]
    if path.startswith("/api/projects"):
        return ["projects"]
    session_paths = (
        "/api/agent/sessions",
        "/api/agent/session",
        "/api/agent/context",
    )
    if any(path == p or path.startswith(p + "/") for p in session_paths):
        return ["sessions"]
    if path.startswith("/api/agent/"):
        return ["settings"]
    return []


def apply_route_tags(app: Any) -> None:
    """Tag every untagged API route by path. Call it after the LAST include.

    Through `iter_api_routes`, not `app.routes`: an included router is one opaque
    node there, so a routed endpoint would silently go untagged.
    """
    for route in iter_api_routes(app):
        if not route.tags:
            route.tags = list(route_tags(route.path))


__all__ = ["API_DESCRIPTION", "OPENAPI_TAGS", "apply_route_tags", "route_tags"]
