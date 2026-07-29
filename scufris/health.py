import os
"""Agent health/diagnostics for the operator console (read-only).

Answers "why won't the agent work?" at a glance: is codex installed and logged in,
is the built-in MCP tool server reachable, are the web assets present, is the agent
enabled. Everything is best-effort - a failing probe becomes a red check, never an
exception - and shell-outs are timeout-guarded so opening the settings page is
cheap even when codex is slow or missing.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

from pydantic import BaseModel

from .config import Settings, canonical_backend
from .sessions import list_sessions, resolve_codex_home

_TIMEOUT = 3.0


class HealthCheck(BaseModel):
    """One diagnostic row. ``status`` is ok | warn | error."""

    name: str
    status: str
    detail: str
    hint: str = ""


class AgentHealth(BaseModel):
    """Read-only diagnostics for the settings/operator console. ``backend`` is the
    effective backend probed (an agent's own backend, or the server default), and
    ``backend_version`` is that backend CLI's reported version - both neutral so a
    claude agent reports claude, not codex."""

    scufris_version: str
    backend: str
    backend_version: str | None = None
    session_count: int = 0
    last_session: datetime | None = None
    checks: list[HealthCheck]


async def _run(cmd: list[str]) -> tuple[int, str]:
    """Run a command with a short timeout; return (returncode, combined output).

    ``(-1, "")`` on timeout / spawn failure - the caller treats that as unknown.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError:
        return -1, ""
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), _TIMEOUT)
    except (asyncio.TimeoutError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return -1, ""
    return proc.returncode or 0, out.decode("utf-8", "replace").strip()


def _scufris_version() -> str:
    try:
        return pkg_version("scufris")
    except PackageNotFoundError:  # pragma: no cover - packaged always has metadata
        return "unknown"


async def agent_health(
    settings: Settings,
    *,
    backend: str | None = None,
    is_orchestrator: bool = True,
    has_scufris_mcp: bool = True,
) -> AgentHealth:
    """Probe the agent's runtime dependencies for the operator console.

    ``backend`` selects which backend CLI to probe (codex/claude/mock); it
    defaults to ``settings.agent_backend`` so the global/orchestrator health is
    unchanged. Pass an agent's own backend to get that agent's diagnostics (a
    claude agent probes the claude CLI, not codex).

    ``is_orchestrator`` + ``has_scufris_mcp`` scope the MCP health rows to the
    AUDIENCE: the orchestrator gets one row per orchestrator server
    (``mcp: scufris`` + ``mcp: den``), a sub-agent gets its callback server
    (``mcp: agent``), and a backend that wires no scufris MCP (opencode/mock,
    ``has_scufris_mcp=False``) gets a single "none" row. Defaults keep the
    orchestrator/global console unchanged.
    """
    effective_backend = canonical_backend(backend or settings.agent_backend)
    checks: list[HealthCheck] = []
    backend_version: str | None = None

    # Agent enabled.
    if settings.agent_enabled:
        checks.append(
            HealthCheck(
                name="agent",
                status="ok",
                detail=f"enabled (backend {effective_backend})",
            )
        )
    else:
        checks.append(
            HealthCheck(
                name="agent",
                status="warn",
                detail="disabled",
                hint="set SCUFRIS_AGENT_ENABLED=1",
            )
        )

    # Backend CLI: probe the binary for the EFFECTIVE backend only (codex or
    # claude). The mock backend needs neither, so nothing is probed there.
    if effective_backend == "codex":
        codex_bin = settings.codex_bin or shutil.which("codex")
        if not codex_bin:
            checks.append(
                HealthCheck(
                    name="codex cli",
                    status="error",
                    detail="not found on PATH",
                    hint="install codex (nixpkgs `codex`)",
                )
            )
        else:
            rc, out = await _run([codex_bin, "--version"])
            backend_version = out or None
            checks.append(
                HealthCheck(
                    name="codex cli",
                    status="ok" if rc == 0 else "warn",
                    detail=out or codex_bin,
                )
            )
            # codex auth (only meaningful once the binary is present).
            rc, out = await _run([codex_bin, "login", "status"])
            logged_in = rc == 0 and "logged in" in out.lower()
            checks.append(
                HealthCheck(
                    name="codex auth",
                    status="ok" if logged_in else "warn",
                    detail=out.splitlines()[0] if out else "unknown",
                    hint="" if logged_in else "run `codex login`",
                )
            )
    elif effective_backend == "claude":
        claude_bin = settings.claude_bin or shutil.which("claude")
        if not claude_bin:
            checks.append(
                HealthCheck(
                    name="claude cli",
                    status="error",
                    detail="not found on PATH",
                    hint="install claude (Claude Code)",
                )
            )
        else:
            rc, out = await _run([claude_bin, "--version"])
            backend_version = out or None
            checks.append(
                HealthCheck(
                    name="claude cli",
                    status="ok" if rc == 0 else "warn",
                    detail=out or claude_bin,
                )
            )
    elif effective_backend == "opencode":
        # opencode is a running daemon, not a CLI binary: probe its HTTP health.
        from .opencode_client import OpencodeClient, OpencodeUnavailable

        client = OpencodeClient(settings.opencode_url, settings.opencode_password)
        try:
            health = await client.health()
            backend_version = health.version or None
            checks.append(
                HealthCheck(
                    name="opencode serve",
                    status="ok" if health.healthy else "warn",
                    detail=f"{settings.opencode_url} (v{health.version})",
                )
            )
        except OpencodeUnavailable as exc:
            checks.append(
                HealthCheck(
                    name="opencode serve",
                    status="error",
                    detail=str(exc),
                    hint=(
                        "start the daemon: "
                        "`OPENCODE_CONFIG=... opencode serve --port 4096`"
                    ),
                )
            )
        finally:
            await client.close()

    # MCP tool servers (in-process probe, one health row PER server for the
    # audience: the orchestrator's scufris + den, or a sub-agent's agent callback
    # server). Same live probe the settings "MCP tools" section groups by, so the
    # Health row and the dropdown never disagree.
    if not settings.agent_tools_enabled:
        checks.append(
            HealthCheck(
                name="mcp tools",
                status="warn",
                detail="disabled",
                hint="set SCUFRIS_AGENT_TOOLS_ENABLED=1",
            )
        )
    elif not has_scufris_mcp:
        checks.append(
            HealthCheck(
                name="mcp tools",
                status="warn",
                detail="none (this backend exposes no scufris tools)",
            )
        )
    else:
        from .mcp_health import probe_server, servers_for_audience

        disabled = set(settings.disabled_tools)
        for server_id, mcp in servers_for_audience(is_orchestrator):
            status, detail, _available, _tools = await probe_server(
                server_id, mcp, disabled
            )
            checks.append(
                HealthCheck(name=f"mcp: {server_id}", status=status, detail=detail)
            )

    # Web assets (the built dashboard the backend serves).
    index = settings.web_dist / "index.html"
    if index.is_file():
        checks.append(
            HealthCheck(name="web assets", status="ok", detail=str(settings.web_dist))
        )
    else:
        checks.append(
            HealthCheck(
                name="web assets",
                status="error",
                detail=f"missing {index}",
                hint="build the frontend (npm run build)",
            )
        )

    # Session summary (cheap, orienting).
    session_count = 0
    last_session: datetime | None = None
    if settings.agent_enabled:
        try:
            sessions = list_sessions(resolve_codex_home(settings), os.getcwd())
            session_count = len(sessions)
            last_session = sessions[0].updated_at if sessions else None
        except Exception:  # noqa: BLE001 - diagnostics never raise
            pass

    return AgentHealth(
        scufris_version=_scufris_version(),
        backend=effective_backend,
        backend_version=backend_version,
        session_count=session_count,
        last_session=last_session,
        checks=checks,
    )
