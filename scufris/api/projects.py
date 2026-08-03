"""Projects: the registered workspaces an agent runs in, and their tatr tasks.

A project is a directory plus a name, and everything that decides whether one is
valid, duplicate or writable lives in `ProjectStore`. What is here is the HTTP
surface over it, plus the discovery view the Projects page is built from -
directories found under the configured base dirs, unioned with what is already
registered.

The two `/projects/...` routes are not API at all: they serve the SPA shell for
the project-detail page, registered ahead of the static mount so a deep link
lands on the shell instead of the static index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from .. import sesh
from ..config import Settings
from ..projects import (
    DuplicateProject,
    InvalidProject,
    Project,
    ProjectNotFound,
    ProjectsReadOnly,
    ProjectStore,
    ProjectTask,
    read_project_tasks,
)
from .models import DeleteResult


class ProjectCreate(BaseModel):
    name: str
    cwd: str
    language: str = ""
    description: str = ""


class ProjectNew(BaseModel):
    """Create a BRAND-NEW project directory under one of the base dirs, then
    register it. `base` must be one of `project_base_dirs` (the endpoint mkdirs
    under it); registering an already-existing dir uses `POST /api/projects`."""

    name: str
    base: str


class DiscoveredProject(BaseModel):
    """A candidate project directory for the Projects page: a discovered dir, a
    registered project, or both. `registered`/`project_id` mark the ones already
    tracked so the UI can offer register vs open."""

    path: str
    name: str
    language: str = ""
    registered: bool = False
    project_id: str | None = None


class DiscoveredProjects(BaseModel):
    """The Projects page payload: the discovered-union-registered directories plus
    the base dirs offered in the create form's picker."""

    projects: list[DiscoveredProject]
    base_dirs: list[str]


class ProjectUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    cwd: str | None = None
    language: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ProjectDeps:
    """What the project routes read: the store, and the settings that say where
    projects may be discovered and created and where the SPA bundle is."""

    settings: Settings
    projects: ProjectStore


def build_project_router(deps: ProjectDeps) -> APIRouter:
    """The project CRUD, the discovery view, the task listing and the SPA shells."""
    router = APIRouter()

    def _project_detail_shell() -> Response:
        """Serve the project-detail SPA shell; the client reads the id from the
        path. Registered before the static mount so `/projects/<id>` routes here
        while `/projects/` (the list) stays on the static index and `/api/...` is
        unaffected. 404 until the frontend is built. Not in the OpenAPI schema."""
        shell = deps.settings.web_dist / "project-detail.html"
        if not shell.is_file():
            raise HTTPException(status_code=404, detail="frontend not built")
        return FileResponse(shell, headers={"Cache-Control": "no-cache"})

    @router.get("/api/projects")
    def list_projects() -> list[Project]:
        """All projects, sorted by name."""
        return deps.projects.list()

    @router.post("/api/projects")
    def create_project(req: ProjectCreate) -> Project:
        """Create a project; 422 for a bad name/cwd, 403 read-only."""
        try:
            return deps.projects.create(
                name=req.name,
                cwd=req.cwd,
                language=req.language,
                description=req.description,
            )
        except ProjectsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (InvalidProject, DuplicateProject) as exc:
            code = 409 if isinstance(exc, DuplicateProject) else 422
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    @router.get("/api/projects/discovered")
    def list_discovered_projects() -> DiscoveredProjects:
        """Directories discovered under the base dirs UNION the registered
        projects, each flagged with whether it is already registered, plus the
        base dirs for the create form's picker - the Projects page's source of
        truth. Declared before `/api/projects/{id}` so "discovered" is not parsed
        as a project id."""
        by_path: dict[str, DiscoveredProject] = {}
        for cand in sesh.discover(deps.settings.project_base_dirs):
            by_path[cand.path] = DiscoveredProject(
                path=cand.path, name=cand.name, language=cand.language
            )
        # Mark discovered dirs that are registered, and ADD registered projects
        # whose cwd is not among the discovered dirs (registered outside a base).
        for project in deps.projects.list():
            key = str(Path(project.cwd).resolve())
            existing = by_path.get(key)
            if existing is not None:
                existing.registered = True
                existing.project_id = project.id
            else:
                by_path[key] = DiscoveredProject(
                    path=key,
                    name=project.name,
                    language=project.language,
                    registered=True,
                    project_id=project.id,
                )
        ordered = sorted(by_path.values(), key=lambda d: (d.name.lower(), d.path))
        base_dirs = [str(b.expanduser()) for b in deps.settings.project_base_dirs]
        return DiscoveredProjects(projects=ordered, base_dirs=base_dirs)

    @router.post("/api/projects/new")
    def create_new_project(req: ProjectNew) -> Project:
        """Make a NEW project directory under an allowed base dir and register it.
        422 for a base outside `project_base_dirs` or an unsafe name, 409 on an id
        collision, 403 read-only."""
        # Guard writability BEFORE the mkdir so a read-only server never has a
        # directory created as a side effect of a refused request.
        if not deps.projects.writable:
            raise HTTPException(
                status_code=403, detail="projects are read-only on this server"
            )
        allowed = {
            str(base.expanduser().resolve()): base.expanduser()
            for base in deps.settings.project_base_dirs
        }
        chosen = allowed.get(str(Path(req.base).expanduser().resolve()))
        if chosen is None:
            raise HTTPException(
                status_code=422,
                detail="base must be one of the configured project base dirs",
            )
        try:
            path = sesh.create(req.name, chosen)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return deps.projects.create(
                name=req.name,
                cwd=str(path),
                language=sesh.infer_language(path),
            )
        except ProjectsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (InvalidProject, DuplicateProject) as exc:
            code = 409 if isinstance(exc, DuplicateProject) else 422
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    @router.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> Project:
        """One project by id; 404 if unknown."""
        try:
            return deps.projects.get(project_id)
        except ProjectNotFound as exc:
            raise HTTPException(status_code=404, detail="no such project") from exc

    @router.patch("/api/projects/{project_id}")
    def update_project(project_id: str, req: ProjectUpdate) -> Project:
        """Update a project's fields; 404 unknown, 422 invalid, 403 read-only."""
        try:
            return deps.projects.update(
                project_id,
                name=req.name,
                cwd=req.cwd,
                language=req.language,
                description=req.description,
            )
        except ProjectsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ProjectNotFound as exc:
            raise HTTPException(status_code=404, detail="no such project") from exc
        except InvalidProject as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/api/projects/{project_id}")
    def delete_project(project_id: str) -> DeleteResult:
        """Delete a project; 404 unknown, 403 read-only."""
        try:
            deps.projects.delete(project_id)
        except ProjectsReadOnly as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ProjectNotFound as exc:
            raise HTTPException(status_code=404, detail="no such project") from exc
        return DeleteResult(deleted=True, current=None)

    @router.get("/api/projects/{project_id}/tasks")
    def get_project_tasks(project_id: str) -> list[ProjectTask]:
        """The project's tatr tasks (its specs); empty when it has no tasks/."""
        try:
            project = deps.projects.get(project_id)
        except ProjectNotFound as exc:
            raise HTTPException(status_code=404, detail="no such project") from exc
        return read_project_tasks(project.cwd)

    @router.get("/projects/{project_id}", include_in_schema=False)
    def project_detail_page(project_id: str) -> Response:
        return _project_detail_shell()

    @router.get("/projects/{project_id}/{rest:path}", include_in_schema=False)
    def project_detail_subpage(project_id: str, rest: str) -> Response:
        return _project_detail_shell()

    return router


__all__ = [
    "DiscoveredProject",
    "DiscoveredProjects",
    "ProjectCreate",
    "ProjectDeps",
    "ProjectNew",
    "ProjectUpdate",
    "build_project_router",
]
