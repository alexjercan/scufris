"""Agents as RECORDS: list, create, read, edit, delete, and what they can reach.

The turn surface is `api/agent_runs.py`; everything here is about the row rather
than about a run. `AgentStore` owns what an agent may be - the name rules, the
project binding, the reserved ids - and this router translates its refusals into
statuses.

The orchestrator is the exception the PATCH route exists to absorb: it has no
`agents` row, so its editable fields live in the SETTINGS store and it reads
back through a synthetic record. The operator's settings form is the same form
either way, which is the whole point of routing that edit here rather than
sending the UI to a second endpoint.

The two `/agents/...` routes are not API: they serve the SPA shell for the
agent-detail page, registered ahead of the static mount so a deep link lands on
the shell instead of the static index.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from ..agent_store import (
    ORCHESTRATOR_ID,
    AgentNotFound,
    AgentRecord,
    AgentsReadOnly,
    AgentStore,
    InvalidAgent,
    ReservedAgent,
)
from ..config import (
    Settings,
    available_backends,
    backend_label,
    canonical_backend,
    default_model_for,
    models_for,
)
from ..enums import AgentState, PermissionMode
from ..orchestrator import AgentRunService
from ..project_capabilities import ProjectCapabilities, read_project_capabilities
from ..settings_store import SettingsReadOnly, SettingsStore, UnknownSettingKey
from .agent_runs import require_agent, require_agent_project
from .models import DeleteResult


class AgentCreate(BaseModel):
    name: str
    project_id: str
    backend: str | None = None
    model: str | None = None
    description: str = ""
    goal: str = ""
    task_id: str = ""
    permission_mode: PermissionMode = PermissionMode.MANUAL


class AgentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    backend: str | None = None
    model: str | None = None
    description: str | None = None
    goal: str | None = None
    task_id: str | None = None
    permission_mode: PermissionMode | None = None


class BackendOption(BaseModel):
    # One selectable backend for the agent create/settings pickers: its id, a
    # friendly label, the default model stamped when it is chosen, and the
    # suggested model catalog (autocomplete; the field still accepts free text).
    id: str
    label: str
    default_model: str
    models: list[str]


class PendingAgent(BaseModel):
    # An agent that needs the orchestrator (BC3): an unacknowledged WAITING
    # (request_input), REPORTED (report_back) or ERROR outcome. ``message`` is the
    # question / result summary / last message.
    agent_id: str
    state: AgentState
    message: str
    run_id: str
    session_id: str | None
    ts: float
    # Who/which orchestrator chat spawned this child (part 3); None = unattributed.
    parent_agent_id: str | None = None
    parent_session_id: str | None = None


@dataclass(frozen=True)
class AgentDeps:
    """What the record routes read.

    ``store`` is the settings store, and it is here for exactly one caller: the
    orchestrator's PATCH, whose config has no row to update. ``runs`` supplies
    the shared agent/project lookups the capabilities route needs.
    """

    settings: Settings
    agents: AgentStore
    store: SettingsStore
    runs: AgentRunService


def build_agent_router(deps: AgentDeps) -> APIRouter:
    """The agent CRUD, the backend picker, the pending poll, the capability view
    and the detail page's SPA shells."""
    router = APIRouter()

    def _agent_detail_shell() -> Response:
        """Serve the agent-detail SPA shell; the client reads the id from the
        path. Registered before the static mount so `/agents/<id>` (and
        `/agents/<id>/settings`) route here while `/agents/` (the list) stays on
        the static index and `/api/...` is unaffected. 404 until the frontend is
        built. Not in the OpenAPI schema (it is a page, not an API)."""
        shell = deps.settings.web_dist / "agent-detail.html"
        if not shell.is_file():
            raise HTTPException(status_code=404, detail="frontend not built")
        return FileResponse(shell, headers={"Cache-Control": "no-cache"})

    def _update_orchestrator(req: AgentUpdate) -> AgentRecord:
        """Apply the orchestrator's editable fields to the SETTINGS store, then
        return the refreshed synthetic record. Name/description/goal/task_id are
        fixed for the orchestrator and ignored. Model follows the EFFECTIVE backend
        (codex -> agent_model, claude -> claude_model, opencode -> opencode_model);
        a blank model re-defaults.
        A backend change clears its session via the store's on_change wiring."""
        updates: dict[str, object] = {}
        eff_backend = canonical_backend(
            req.backend if req.backend is not None else deps.settings.agent_backend
        )
        if req.backend is not None:
            updates["agent_backend"] = req.backend
        if req.model is not None:
            model = req.model.strip() or default_model_for(deps.settings, eff_backend)
            key = {
                "claude": "claude_model",
                "opencode": "opencode_model",
            }.get(eff_backend, "agent_model")
            updates[key] = model
        if req.permission_mode is not None:
            updates["agent_permission_mode"] = req.permission_mode
        if updates:
            try:
                deps.store.apply(updates)
            except SettingsReadOnly as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except (UnknownSettingKey, ValidationError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return deps.agents.get(ORCHESTRATOR_ID)

    @router.get("/api/agents")
    def list_agents() -> list[AgentRecord]:
        """All configured agents, sorted by name."""
        return deps.agents.list()

    @router.post("/api/agents")
    def create_agent(req: AgentCreate) -> AgentRecord:
        """Create an agent bound to a project; 422 bad field/unknown project,
        403 read-only."""
        try:
            return deps.agents.create(
                name=req.name,
                project_id=req.project_id,
                backend=req.backend,
                model=req.model,
                description=req.description,
                goal=req.goal,
                task_id=req.task_id,
                permission_mode=req.permission_mode,
            )
        except AgentsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except InvalidAgent as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/api/agents/backends")
    def list_agent_backends() -> list[BackendOption]:
        """The backends an agent may use (mock only when the dev flag is on),
        each with its friendly label and default model, so the create/settings
        pickers are server-authoritative. Declared before /api/agents/{id} so
        "backends" is not parsed as an agent id."""
        return [
            BackendOption(
                id=b,
                label=backend_label(b),
                default_model=default_model_for(deps.settings, b),
                models=models_for(deps.settings, b),
            )
            for b in available_backends(deps.settings)
        ]

    @router.get("/api/agents/pending")
    def list_pending_agents(
        parent_session_id: str | None = None,
    ) -> list[PendingAgent]:
        """The agents that need the orchestrator (BC3): those with an
        unacknowledged needs-input (WAITING, from request_input), reported-done
        (REPORTED, from report_back) or ERROR outcome, newest first. The
        orchestrator polls this to find blocked or finished sub-agents. Declared
        before /api/agents/{id} (like /api/agents/backends) so "pending" is not
        parsed as an agent id.

        ``parent_session_id`` scopes to one orchestrator chat (part 3): the result
        keeps children that chat spawned PLUS unattributed ones (UI-launched, or
        spawned before a fresh turn had a session), and drops children clearly
        owned by a DIFFERENT chat - so a poll from chat A never sees chat B's
        children, and nothing is orphaned. Each row is annotated with its parent."""
        pending = deps.agents.pending_outcomes()
        rows: list[PendingAgent] = []
        for agent_id, o in pending.items():
            parent_agent, parent_sess = deps.agents.parent_of(agent_id)
            # Scope: keep this chat's own children and unattributed ones; drop
            # another chat's. No filter (None query) -> keep all (back-compat).
            if (
                parent_session_id is not None
                and parent_sess is not None
                and parent_sess != parent_session_id
            ):
                continue
            rows.append(
                PendingAgent(
                    agent_id=agent_id,
                    state=o.state,
                    message=o.message,
                    run_id=o.run_id,
                    session_id=o.session_id,
                    ts=o.ts,
                    parent_agent_id=parent_agent,
                    parent_session_id=parent_sess,
                )
            )
        rows.sort(key=lambda r: r.ts, reverse=True)
        return rows

    @router.get("/api/agents/{agent_id}")
    def get_agent(agent_id: str) -> AgentRecord:
        """One agent by id; 404 if unknown."""
        try:
            return deps.agents.get(agent_id)
        except AgentNotFound as exc:
            raise HTTPException(status_code=404, detail="no such agent") from exc

    @router.patch("/api/agents/{agent_id}")
    def update_agent(agent_id: str, req: AgentUpdate) -> AgentRecord:
        """Update an agent's config; 404 unknown, 422 invalid, 403 read-only.

        The orchestrator has no ``agents`` row - its config lives in the settings
        store - so its edits (backend/model/permission_mode) route THERE and it
        reads them back through the synthetic record. Every other agent updates its
        own record. The unified settings form (U3) is identical either way."""
        if agent_id == ORCHESTRATOR_ID:
            return _update_orchestrator(req)
        try:
            return deps.agents.update(
                agent_id,
                name=req.name,
                backend=req.backend,
                model=req.model,
                description=req.description,
                goal=req.goal,
                task_id=req.task_id,
                permission_mode=req.permission_mode,
            )
        except AgentsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ReservedAgent as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except AgentNotFound as exc:
            raise HTTPException(status_code=404, detail="no such agent") from exc
        except InvalidAgent as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/api/agents/{agent_id}")
    def delete_agent(agent_id: str) -> DeleteResult:
        """Delete an agent; 404 unknown, 403 read-only or reserved."""
        try:
            deps.agents.delete(agent_id)
        except AgentsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ReservedAgent as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except AgentNotFound as exc:
            raise HTTPException(status_code=404, detail="no such agent") from exc
        return DeleteResult(deleted=True, current=None)

    @router.get("/api/agents/{agent_id}/capabilities")
    def get_agent_capabilities(agent_id: str) -> ProjectCapabilities:
        """The read-only skills + custom tools THIS agent's PROJECT defines in its
        working tree - its ``.claude/skills`` / ``.mcp.json`` (claude) or
        ``.codex`` equivalents (codex), discovered PROVIDER-aware from the agent's
        backend. What the settings page surfaces so the operator can see the
        recipes and tools an agent can be steered toward. The orchestrator (and any
        project-less agent) has no project tree -> an empty set. 404 unknown agent;
        nothing here is writable or executed."""
        agent = require_agent(deps.runs, agent_id)
        project = require_agent_project(deps.runs, agent)
        if project is None:
            return ProjectCapabilities()
        return read_project_capabilities(project.cwd, canonical_backend(agent.backend))

    @router.get("/agents/{agent_id}", include_in_schema=False)
    def agent_detail_page(agent_id: str) -> Response:
        return _agent_detail_shell()

    @router.get("/agents/{agent_id}/{rest:path}", include_in_schema=False)
    def agent_detail_subpage(agent_id: str, rest: str) -> Response:
        return _agent_detail_shell()

    return router


__all__ = [
    "AgentCreate",
    "AgentDeps",
    "AgentUpdate",
    "BackendOption",
    "PendingAgent",
    "build_agent_router",
]
