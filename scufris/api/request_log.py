"""The per-request log line.

The app's other middleware, registered outside the auth gate so a denial is
still logged. It lives here rather than in `logsetup` because it is HTTP - it
reads a Starlette request and a response status - while `logsetup` is imported
by the MCP servers and the agent subprocesses, which have no web stack.

The request id itself comes from `logsetup`: this middleware only decides when a
new one starts.
"""

from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable

from fastapi import Request, Response

from ..logsetup import new_request_id, set_request_id

logger = logging.getLogger(__name__)


async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Tag each request with an id and log method/path/status/duration.

    At DEBUG so `--debug` shows every request without the default INFO being
    flooded by the dashboard's 2s stats/processes polling; 5xx at WARNING.
    """
    set_request_id(new_request_id())
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    level = logging.WARNING if response.status_code >= 500 else logging.DEBUG
    logger.log(
        level,
        "%s %s -> %d in %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response
