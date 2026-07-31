"""Shared, dependency-light helpers for the Scufris MCP servers.

The scufris MCP surface is split across three single-audience servers -
``mcp_server`` (orchestrator agentic), ``den_mcp_server`` (the operator's
journal + macros life tools) and ``agent_mcp_server`` (the sub-agent callback
tools). This module holds the pieces they all need: the curated-command shell
wrapper (``_run``), the dashboard HTTP bridge (``_api_call``), and the
operator disabled-tools filter applied at startup.

It deliberately imports NOTHING heavy (no psutil, no agent store, no FastMCP at
import time) so ``den_mcp_server`` can reuse it while staying reusable on a box
that has only the ``today``/``macros`` CLIs. FastMCP is referenced only as a
type in ``apply_disabled_tools`` under ``TYPE_CHECKING``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Cap tool output so a huge result can't blow up the model context.
_MAX_OUTPUT = 20_000
_TIMEOUT_SECONDS = 15.0
_API_TIMEOUT = 15.0


def _run(args: list[str], *, timeout: float = _TIMEOUT_SECONDS) -> str:
    """Run a curated command safely and return bounded combined output.

    `shell=False` with an explicit argument list; the executable is resolved on
    PATH; failures and timeouts are reported as text rather than raised, so the
    agent gets a usable message.
    """
    exe = shutil.which(args[0])
    if exe is None:
        logger.info("run %s: not found on PATH", args[0])
        return f"error: {args[0]} not found on PATH"
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [exe, *args[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.info("run %s: timed out after %ss", args[0], timeout)
        return f"error: {args[0]} timed out after {timeout}s"
    output = proc.stdout
    if proc.returncode != 0:
        logger.info("run %s: exit=%d", args[0], proc.returncode)
        output = (output + "\n" + proc.stderr).strip() or f"exit {proc.returncode}"
    logger.debug(
        "run %s -> exit=%d bytes=%d in %.2fs",
        " ".join(args),
        proc.returncode,
        len(output),
        time.monotonic() - started,
    )
    return output[:_MAX_OUTPUT]


def _api_base() -> str:
    """Base URL of the dashboard's HTTP API (``SCUFRIS_API_BASE``, injected by the
    dashboard when it spawns a server); defaults to the usual local bind."""
    import os

    return os.environ.get("SCUFRIS_API_BASE", "http://127.0.0.1:8000").rstrip("/")


# The machine credential for the dashboard's own API, when a tool runs IN the
# dashboard process (the operator tool console) rather than in an MCP subprocess.
# A ContextVar rather than a module global so two apps in one process - which the
# test suite does - cannot clobber each other's token, and so the value is scoped
# to the call rather than ambient. `asyncio.to_thread` copies the context, so it
# survives the console's off-loop hop. Review round 1, findings 2 and 3.
api_token_var: ContextVar[str] = ContextVar("scufris_api_token", default="")


def _api_headers() -> dict[str, str]:
    """Credentials for the dashboard's own API.

    Two carriers, one meaning. An MCP SUBPROCESS reads ``SCUFRIS_API_TOKEN`` from
    the env the dashboard injected into that server specifically; an IN-PROCESS
    tool run reads the ContextVar the console set. With neither (a bare
    ``scufris mcp-server`` run with no dashboard behind it) the call goes out
    unauthenticated and a gated dashboard refuses it with a 401 the model can
    read - the honest outcome, not a silent bypass.
    """
    import os

    token = api_token_var.get() or os.environ.get("SCUFRIS_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _api_call(
    method: str,
    path: str,
    *,
    body: object | None = None,
    timeout: float = _API_TIMEOUT,
    read_unbounded: bool = False,
) -> str:
    """Call the local dashboard API and return bounded text (never raises).

    Failures and non-2xx responses come back as ``error: ...`` text, like ``_run``,
    so the model gets a usable message instead of an exception. Output is truncated
    to ``_MAX_OUTPUT``.

    ``read_unbounded`` disables the READ timeout for callers that stream a full
    agent turn (``message_agent``): the sub-agent turn self-terminates (its
    runner's idle guard and the supervisor heartbeat bound it), so the
    orchestrator must not cut a long-but-progressing turn on a wall-clock read
    cap - the same idle-vs-wall-clock fix as the codex runner.
    ``timeout`` still bounds connect/write/pool so an unreachable API fails fast.
    """
    import httpx

    url = _api_base() + path
    bound: float | httpx.Timeout = (
        httpx.Timeout(timeout, read=None) if read_unbounded else timeout
    )
    try:
        resp = httpx.request(
            method, url, json=body, timeout=bound, headers=_api_headers()
        )
    except httpx.HTTPError as exc:
        logger.info("api %s %s: %s", method, path, exc)
        return f"error: request to {path} failed: {exc}"
    if resp.status_code >= 400:
        detail = resp.text.strip()
        logger.info("api %s %s -> %d", method, path, resp.status_code)
        return f"error: {resp.status_code} from {path}: {detail[:500]}"
    return resp.text[:_MAX_OUTPUT]


def _disabled_tools() -> list[str]:
    """Tool names the operator has disabled, from ``SCUFRIS_DISABLED_TOOLS``.

    The dashboard injects this env (comma-separated) when it spawns a server,
    from the runtime-editable ``disabled_tools`` setting. Only the orchestrator
    servers carry it; the sub-agent callback server ignores it.
    """
    import os

    raw = os.environ.get("SCUFRIS_DISABLED_TOOLS", "")
    return [name.strip() for name in raw.split(",") if name.strip()]


def apply_disabled_tools(mcp: FastMCP, names: list[str]) -> list[str]:
    """Remove ``names`` from ``mcp``'s live tool registry; return those removed.

    Done before the server serves any request, so a disabled tool is never
    advertised or callable - enforcement lives here, not in the UI.
    """
    removed: list[str] = []
    for name in names:
        if mcp._tool_manager.get_tool(name) is not None:
            mcp._tool_manager.remove_tool(name)
            removed.append(name)
    return removed
