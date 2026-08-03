"""Bridging `Settings` into the environment variables an in-process tool reads.

Two of the operator console's paths run an MCP tool inside THIS process rather
than in the agent's MCP subprocess, and the tools read their configuration from
`os.environ` (`mcp_server._api_base`, `mcp_server._den_path`). The agent path
injects that env into the subprocess; the in-process path has to bridge it here.

Both use `setdefault`, so an explicit operator override always wins.

Its own module, not a private helper on `scufris.app`: `ensure_den_path` is
called from two routers and the Telegram wiring, and `ensure_api_base` from
`run_server`, and leaving them in the application factory would make every one
of those callers import the factory that imports them.
"""

from __future__ import annotations

import os

from .config import Settings


def ensure_api_base(settings: Settings) -> str:
    """Default ``SCUFRIS_API_BASE`` to THIS dashboard's own base, so an in-process
    tool run (the operator console's ``/api/agent/tools/{name}/run``) loops back to
    this server rather than ``mcp_server._api_base``'s hardcoded ``:8000`` default -
    which, on a non-8000 port, silently hits a different (often stale) instance.

    ``setdefault`` so an explicit operator override wins (a non-loopback
    deployment). ``127.0.0.1`` rather than ``settings.host`` because the host may
    be ``0.0.0.0`` (bind-all), which is not a connectable address. Returns the
    effective base."""
    return os.environ.setdefault(
        "SCUFRIS_API_BASE", f"http://127.0.0.1:{settings.port}"
    )


def ensure_den_path(settings: Settings) -> None:
    """Bridge ``settings.den_path`` into ``SCUFRIS_DEN_PATH`` for an IN-PROCESS tool
    run (the operator console's ``/api/agent/tools/{name}/run``), so the ``journal_*``
    tools resolve the den the same way they do in an agent turn.

    The journal tools read ``SCUFRIS_DEN_PATH`` from the environment
    (``mcp_server._den_path``), which the agent path injects into the MCP SUBPROCESS
    env. The console runs the tool in THIS process instead, and pydantic loads
    ``den_path`` from ``.env`` into the ``Settings`` object WITHOUT exporting it to
    ``os.environ`` - so without this bridge the console sees an unset var and reports
    "not configured". Mirrors ``ensure_api_base``. ``setdefault`` so an explicit env
    (the deployed service sets ``SCUFRIS_DEN_PATH`` directly) wins; a no-op when
    ``den_path`` is unset (the tools stay correctly inert). Isolation is unaffected:
    a sub-agent cannot call ``journal_*`` at all (the ``den`` server is never
    registered on a sub-agent turn), so a subprocess inheriting the var is moot."""
    if settings.den_path is not None:
        os.environ.setdefault("SCUFRIS_DEN_PATH", str(settings.den_path))


__all__ = ["ensure_api_base", "ensure_den_path"]
