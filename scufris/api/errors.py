"""Turning a service's refusals into statuses a client can act on.

Its own module rather than a private helper on one router: both the host router
and anything else that talks to `hostd` must map the same error the same way, and
a second copy of this table is how a 409 quietly becomes a 502 on one surface.
The orchestrator services are here for the same reason - the turn routes, the
per-agent lifecycle routes and anything that grows on top of them all speak one
mapping.
"""

from __future__ import annotations

from fastapi import HTTPException

from scufris_hostd import ErrorCode

from ..hostclient import HostdError, HostdUnavailable
from ..orchestrator.errors import (
    AgentDisabled,
    AgentProjectMissing,
    NoActiveRun,
    OrchestratorError,
    RunAlreadyActive,
    TurnEndedWithoutReply,
    TurnFailed,
)

#: The helper's refusal codes, as the statuses a client can distinguish. Anything
#: not listed - and any exception that is not the helper's - is a 502: the app
#: reached something it does not understand, which is a gateway problem.
_HOSTD_STATUS: dict[ErrorCode, int] = {
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.EXPIRED: 409,
    ErrorCode.DRIFTED: 409,
    ErrorCode.ALREADY_USED: 409,
    ErrorCode.REFUSED: 422,
    ErrorCode.BAD_REQUEST: 422,
    ErrorCode.UNAUTHORIZED: 502,
}


def hostd_http_error(exc: Exception) -> HTTPException:
    """Map the helper's own refusals onto statuses a client can act on."""
    if isinstance(exc, HostdUnavailable):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, HostdError):
        return HTTPException(
            status_code=_HOSTD_STATUS.get(exc.code, 502), detail=exc.detail
        )
    return HTTPException(status_code=502, detail=str(exc))


#: The turn path's refusals, as the statuses the dashboard and the agent tools
#: already act on. Exact classes, not an isinstance walk: a new refusal must be
#: given a status here rather than inheriting a neighbour's by accident.
_ORCHESTRATOR_STATUS: dict[type[OrchestratorError], int] = {
    RunAlreadyActive: 409,
    NoActiveRun: 404,
    AgentDisabled: 503,
    AgentProjectMissing: 422,
    TurnFailed: 503,
    TurnEndedWithoutReply: 500,
}


def orchestrator_http_error(exc: OrchestratorError) -> HTTPException:
    """Map an orchestrator refusal onto the status its clients expect.

    An unlisted refusal is a 500: it is a condition this table has not been
    taught, which is a server problem and not something the caller can fix.
    """
    return HTTPException(
        status_code=_ORCHESTRATOR_STATUS.get(type(exc), 500), detail=exc.detail
    )


__all__ = ["hostd_http_error", "orchestrator_http_error"]
