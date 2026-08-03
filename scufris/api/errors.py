"""Turning the root helper's refusals into statuses a client can act on.

Its own module rather than a private helper on one router: both the host router
and anything else that talks to `hostd` must map the same error the same way, and
a second copy of this table is how a 409 quietly becomes a 502 on one surface.
"""

from __future__ import annotations

from fastapi import HTTPException

from ..hostclient import HostdError, HostdUnavailable
from ..hostd.protocol import ErrorCode

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


__all__ = ["hostd_http_error"]
