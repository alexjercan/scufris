"""The NixOS configuration-change surface.

The configuration repository is a PROJECT: an agent edits and commits it through
the ordinary project machinery, and none of that happens here. What happens here
is the last mile - resolve a ref to a commit, build it as the operator, and hand
the resulting store path to the helper as an `activate` proposal - and all of
that lives in `ConfigChangeService`. These routes name who is asking, call it,
and translate its refusals.

Its own module rather than a section of `host.py`: the two together are over the
600-line source cap, and they are separable - this one talks to the build
registry, that one to the root helper.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scufris_hostctl import (
    ChangeInFlight,
    ConfigChange,
    ConfigChangeRefused,
    ConfigChangeService,
    NoRunningBuild,
    UnknownChange,
)

from .auth import SessionGate
from .sse import last_event_id, relay_bus_sse


class ConfigChangeRequest(BaseModel):
    """Build a committed configuration and propose activating it.

    A REF, never a store path. Which revision gets built is a caller's to name -
    it is a commit in a repository, reviewable in git - but what that revision
    BUILDS INTO is resolved by this server, because a caller-supplied store path
    would mean the model choosing what gets activated.
    """

    # Empty means the configured host config repo. A path may be a linked
    # worktree - which is where an agent will have been working.
    repo: str = ""
    # Empty means HEAD of that working tree.
    ref: str = ""
    # Which nixosConfiguration; empty means the configured one, then the hostname.
    attr: str = ""
    agent: str = ""
    run: str = ""


@dataclass(frozen=True)
class HostConfigDeps:
    """The change flow, and the gate that says who is asking for one."""

    gate: SessionGate
    changes: ConfigChangeService


def build_hostconfig_router(deps: HostConfigDeps) -> APIRouter:
    """The `/api/host/config/changes` routes over an explicit change service."""
    router = APIRouter()

    async def _change_or_404(change_id: str) -> ConfigChange:
        try:
            return await deps.changes.get(change_id)
        except UnknownChange:
            raise HTTPException(
                status_code=404, detail="no such config change"
            ) from None

    @router.post("/api/host/config/changes", status_code=201)
    async def propose_config_change(
        body: ConfigChangeRequest, request: Request
    ) -> ConfigChange:
        """Build a committed configuration and propose activating it.

        Open to an agent, like every other propose path: building changes nothing
        and the activation it produces still needs the operator. The build runs as
        this process's user - never root - and reads the tree from the COMMIT, so
        this endpoint cannot touch the repository's working tree.
        """
        requester = await deps.gate.requester_identity(
            request, agent=body.agent, run=body.run
        )
        try:
            return await deps.changes.start(
                requester, ref=body.ref, repo=body.repo, attr=body.attr
            )
        except ConfigChangeRefused as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ChangeInFlight as exc:
            # Refused, not queued: two builds of one repository contend for the
            # same evaluation and store, and a queued NixOS build sits for an
            # hour with no visible reason.
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/api/host/config/changes")
    async def list_config_changes() -> list[ConfigChange]:
        """Configuration changes, newest first."""
        return await deps.changes.list()

    @router.get("/api/host/config/changes/{change_id}")
    async def get_config_change(change_id: str) -> ConfigChange:
        return await _change_or_404(change_id)

    @router.get("/api/host/config/changes/{change_id}/events")
    async def config_change_events(
        change_id: str, http_request: Request
    ) -> StreamingResponse:
        """Relay a build's live log as SSE."""
        change = await _change_or_404(change_id)
        bus = deps.changes.bus(change)
        if bus is None:
            raise HTTPException(status_code=404, detail="this change has no live build")
        return relay_bus_sse(bus, last_event_id(http_request))

    @router.post("/api/host/config/changes/{change_id}/cancel")
    async def cancel_config_change(change_id: str) -> ConfigChange:
        """Stop a build that is running.

        Not operator-only: a build holds no privilege and stopping it undoes
        nothing. What IS operator-only is approving the activation it produces.
        """
        try:
            return await deps.changes.cancel(change_id)
        except UnknownChange:
            raise HTTPException(
                status_code=404, detail="no such config change"
            ) from None
        except NoRunningBuild as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return router


__all__ = [
    "ConfigChangeRequest",
    "HostConfigDeps",
    "build_hostconfig_router",
]
