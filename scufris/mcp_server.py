"""Scufris `scufris` MCP server: the orchestrator's agentic tools.

Exposed over stdio (MCP) and registered with a backend per-invocation by the
agent (no `~/.codex` edits). The allowlist IS this set of handlers - there is no
generic "run any command" tool. Each tool that shells out uses a fixed argument
list (never a shell string), a timeout, and bounded output.

This is ONE of three single-audience scufris MCP servers (see
``tasks/20260727-105609/DECISION.md``): ``scufris`` (this module, orchestrator
agentic), ``den`` (``den_mcp_server``, the operator's journal + macros life
tools) and ``agent`` (``agent_mcp_server``, the sub-agent callback tools). Only
an ORCHESTRATOR turn registers this server, so its tools are never advertised to
a sub-agent (the guarantee is "not registered", not a runtime filter).

The surface here: read-only host introspection - the light live snapshot
(host_stats, disk_usage, list_processes) plus the deep inspection over
``scufris.host`` (host_units, host_failed_units, host_unit_status, host_journal,
host_storage, host_largest_directories, host_reclaimable_space, host_network,
host_thermal, host_what_provides, host_generation_diff, host_flake_status) -
the host ACTION tools (propose_host_action, host_action_status,
host_action_audit), which can only ever ASK for a privileged change: there is no
approve tool here and there will not be one, because approving is an operator
act gated on a real session (see AGENTS.md, privileged host actions) -
read-only agent observation (list_agents, agent_status), the
orchestrator CONTROL tools that call the dashboard's own HTTP API - full CRUD
over projects (list/get/create/update/delete) and agents (create/update/delete +
run/message), where the write tools edit REGULAR agents only (the orchestrator
configures itself via settings) - and the escalation-inbox tools (pending_agents,
acknowledge) that surface sub-agents' callbacks. tatr task management is
intentionally NOT here: the orchestrator runs the `tatr` skill via Bash, so a
dedicated MCP wrapper would be redundant.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING
from urllib.parse import quote

from mcp.server.fastmcp import FastMCP

from .mcp_common import (
    _MAX_OUTPUT,
    _api_call,
    _disabled_tools,
    _run,
    apply_disabled_tools,
)
from .metrics import PsutilCollector
from .processes import ProcessList, PsutilProcessCollector

if TYPE_CHECKING:
    # Imported lazily inside the tool helpers (to keep the MCP server's startup
    # import light); named here only for type checking.
    from .agent_store import AgentStore
    from .config import Settings
    from .host import HostInspector, Scope

logger = logging.getLogger(__name__)

mcp = FastMCP("scufris")
_collector = PsutilCollector()
# Primed at import so per-process cpu% is a real delta on the first sample
# (psutil.process_iter reuses Process objects internally).
_proc_collector = PsutilProcessCollector()


def _human_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def _format_processes(plist: ProcessList, limit: int) -> str:
    """Render the top application groups as a compact fixed-width table."""
    header = f"{'APPLICATION':<24} {'CPU%':>6} {'MEM':>8} {'#':>4}"
    lines = [header, f"total processes: {plist.total}"]
    for group in plist.groups[: max(1, limit)]:
        lines.append(
            f"{group.name[:24]:<24} {group.cpu_percent:>6.1f} "
            f"{_human_bytes(group.mem_rss):>8} {group.count:>4}"
        )
    return "\n".join(lines)


@mcp.tool()
def host_stats() -> dict[str, object]:
    """Complete structured snapshot of THIS host's live metrics as JSON: CPU model
    and per-core load, memory, swap, disks, network throughput, load average,
    uptime, and GPUs.

    This is the PREFERRED, authoritative way to answer any question about the
    host's current state, hardware, or resource usage. Call this FIRST and use its
    result instead of shell commands like `uname`, `lscpu`, `top`, `free`,
    `uptime`, `nvidia-smi`, or reading `/proc` - one call replaces piecing together
    shell output, and it is curated for this exact host.
    """
    return _collector.sample().model_dump(mode="json")


@mcp.tool()
def disk_usage() -> str:
    """Disk usage per real filesystem (like `df -h`), excluding tmpfs/overlay noise.

    PREFER this over running `df` yourself in the shell for any "how full are my
    disks" question - it is already filtered to the real filesystems on this host.
    """
    return _run(
        [
            "df",
            "-h",
            "-x",
            "tmpfs",
            "-x",
            "devtmpfs",
            "-x",
            "squashfs",
            "-x",
            "overlay",
        ]
    )


@mcp.tool()
def list_processes(limit: int = 15) -> str:
    """Top running applications by CPU, grouped by name (like a compact htop).

    PREFER this over shell `ps`/`top`/`htop` for any "what is using CPU/memory"
    question - it is already grouped and ranked for this host.
    """
    return _format_processes(_proc_collector.sample(), limit)


# --- host inspection (read-only) ---------------------------------------------
#
# The deep read-only surface over this NixOS box: units, journal, storage,
# network, thermals, packages and generations (task 20260729-125024, epic
# 20260729-124655). Everything runs through `scufris.host`, so each tool is
# bounded, classified and honest about what it could not read - see that
# package's docstring. Nothing here mutates the system: the mutating half is a
# separate, approval-gated surface: see the host ACTION tools below, which
# propose rather than act.


def _inspector() -> "HostInspector":
    """A HostInspector configured from settings.

    Built per call rather than at import: this module is imported by the
    dashboard as well as run as an MCP subprocess, and `Settings()` at import
    time would freeze whatever the environment looked like then.
    """
    from .config import Settings
    from .host import HostInspector

    settings = Settings()
    return HostInspector(config_repo=settings.host_config_repo)


def _scope(value: str) -> "Scope | None":
    """Parse a scope argument, or None when it is not a scope.

    A wrong scope is a real fork in the answer (a user unit and a system unit can
    share a name), so it is refused rather than defaulted.

    None rather than an error STRING as the failure signal: `Scope` is a StrEnum,
    so a returned scope is itself a `str` and an `isinstance(x, str)` check at the
    call site would classify every valid scope as an error. Caught by a test that
    passed a good scope and got the error branch.
    """
    from .host import Scope

    text = (value or "system").strip().lower()
    return Scope(text) if text in {s.value for s in Scope} else None


def _bad_scope(value: str) -> str:
    return f"error: scope must be 'system' or 'user', not {value!r}"


@mcp.tool()
def host_units(state: str = "", pattern: str = "", scope: str = "system") -> str:
    """List this host's systemd units, optionally filtered by state or name.

    PREFER this over shell `systemctl list-units` - it parses systemd's JSON
    output, is bounded, and says explicitly when nothing matched instead of
    returning a blank you might read as "all fine".

    `state` is a systemd state word (failed, running, active, inactive), `pattern`
    a unit glob like "nginx*", and `scope` is "system" (pid 1) or "user" (the
    operator's session units - scufris itself is a USER unit on this host).
    """
    from .host import render

    parsed = _scope(scope)
    if parsed is None:
        return _bad_scope(scope)
    return render.render_units(
        _inspector().list_units(scope=parsed, state=state, pattern=pattern)
    )


@mcp.tool()
def host_failed_units(scope: str = "system") -> str:
    """The systemd units currently in the FAILED state on this host.

    Call this FIRST for "did anything break", "is anything broken", or "did
    anything fail overnight". PREFER it over shell `systemctl --failed`. An empty
    result is stated as "nothing is in a failed state", so you can report that
    with confidence rather than guessing from silence.

    Check BOTH scopes when the question is open-ended: "system" misses the
    operator's user units (scufris runs as one).
    """
    from .host import render

    parsed = _scope(scope)
    if parsed is None:
        return _bad_scope(scope)
    return render.render_units(_inspector().failed_units(scope=parsed))


@mcp.tool()
def host_unit_status(name: str, scope: str = "system") -> str:
    """One systemd unit's state, last result, restart count and memory use.

    PREFER this over shell `systemctl status <unit>` for "is X running", "why did
    X restart", or "when did X last start". Says so explicitly when no such unit
    is loaded, rather than printing an empty status block.
    """
    from .host import render

    parsed = _scope(scope)
    if parsed is None:
        return _bad_scope(scope)
    return render.render_unit_status(_inspector().unit_status(name, scope=parsed))


@mcp.tool()
def host_journal(
    unit: str = "",
    priority: str = "",
    since: str = "1 hour ago",
    until: str = "",
    lines: int = 50,
    scope: str = "system",
) -> str:
    """Read a BOUNDED window of this host's systemd journal.

    PREFER this over shell `journalctl` - it caps both the line count and the
    byte size, so a chatty unit cannot flood your context, and it marks
    truncation explicitly.

    `unit` limits to one unit, `priority` to that severity and worse (emerg,
    alert, crit, err, warning, notice, info, debug), `since`/`until` take
    journalctl's own time words ("30 min ago", "yesterday", "2026-07-29 12:00").
    Narrow with `unit` and `since` before raising `lines`: a wide window returns
    the truncation marker, not more data.
    """
    from .host import render

    parsed = _scope(scope)
    if parsed is None:
        return _bad_scope(scope)
    return render.render_journal(
        _inspector().journal(
            unit=unit,
            scope=parsed,
            priority=priority,
            since=since,
            until=until,
            lines=lines,
        )
    )


@mcp.tool()
def host_storage() -> str:
    """Filesystem usage, the Nix store's filesystem, and the NixOS generations.

    Call this FIRST for "how full is the disk" or as the opening move of "what
    filled the disk". PREFER it over shell `df` / `nixos-rebuild
    list-generations`. Follow up with `host_largest_directories` to find WHAT
    filled a specific filesystem, and `host_reclaimable_space` for how much a
    garbage collection would remove.
    """
    from .host import render

    return render.render_storage(_inspector().storage())


@mcp.tool()
def host_largest_directories(root: str, depth: int = 1, limit: int = 20) -> str:
    """The biggest directories under `root`, to answer "what filled the disk".

    PREFER this over shell `du` - it stays on `root`'s filesystem (so pointing it
    at "/" does not wander into the Nix store), bounds its depth and its output,
    and reports partial results with a caveat when some directories are
    unreadable.

    This WALKS the directory tree and can take tens of seconds on a large one.
    Start with depth 1 on a specific root ("/home/alex", "/var"), then drill into
    the biggest entry, rather than asking for depth 3 on "/".
    """
    from .host import render

    return render.render_largest_directories(
        _inspector().largest_directories(root, depth=depth, limit=limit)
    )


@mcp.tool()
def host_reclaimable_space() -> str:
    """How many Nix store paths are dead and could be garbage-collected.

    PREFER this over shell `nix-collect-garbage --dry-run`. Read-only by
    construction: it runs `nix-store --gc --print-dead`, which only ENUMERATES.
    Actually freeing the space, or trimming generations by age, is a privileged
    operator-approved action this read-only toolset cannot perform - do not
    offer to run it.

    Note what it returns: a path COUNT, not a byte total, so do not present the
    number as an amount of disk space. This WALKS the whole store and can take a
    minute.
    """
    from .host import render

    return render.render_reclaimable(_inspector().reclaimable_space())


@mcp.tool()
def host_network() -> str:
    """Network interfaces, what is listening, and the firewall this host declares.

    PREFER this over shell `ip addr` / `ss -tlnp` / `iptables -L` for "what is
    listening", "what is exposed", or "is port N open".

    Two honesty notes you should carry into your answer: socket owners are only
    visible for the operator's own processes (others are marked so), and the
    firewall shown is the one the current NixOS generation DECLARES - the live
    iptables table needs root and is not readable here.
    """
    from .host import render

    return render.render_network(_inspector().network())


@mcp.tool()
def host_thermal() -> str:
    """Temperatures, thermal THROTTLING counters, fans and battery for this host.

    Call this FIRST for "why is it hot", "is it throttling", or "is it running
    warm". PREFER it over shell `sensors` (not installed here) or reading
    /sys by hand.

    The throttle counters are the part that actually settles the question: they
    are cumulative kernel records of the CPU being held back, so they show
    throttling that already happened even when the current temperature looks
    fine. A temperature alone cannot tell you that.
    """
    from .host import render

    return render.render_thermal(_inspector().thermal())


@mcp.tool()
def host_what_provides(binary: str) -> str:
    """Which Nix package provides a command on this host.

    PREFER this over shell `which` + `readlink` for "where does X come from" or
    "what package is X in" - it follows the symlink chain into the store and
    reports the package name and version.
    """
    from .host import render

    return render.render_provider(_inspector().what_provides(binary))


@mcp.tool()
def host_generation_diff(before: int = 0, after: int = 0) -> str:
    """What changed between two NixOS system generations (a closure diff).

    PREFER this over shell `nix store diff-closures`. With no arguments it
    compares the previous generation to the current one, answering "what did the
    last rebuild change". Pass generation NUMBERS (from `host_storage`) to
    compare any two.

    When the two closures are identical this says so explicitly - nix itself
    prints nothing at all in that case, so an empty result there means "no
    change", never "the command failed".
    """
    from .host import render

    inspector = _inspector()
    if before <= 0 or after <= 0:
        generations = inspector.generations()
        numbers = [g.number for g in generations.generations]
        if not generations.ok:
            return f"error: {generations.available.reason}"
        if len(numbers) < 2:
            return (
                "this host has fewer than two generations, so there is nothing to diff"
            )
        before, after = numbers[1], numbers[0]
    return render.render_closure_diff(inspector.closure_diff(before, after))


@mcp.tool()
def host_flake_status() -> str:
    """How old the pinned inputs of this host's NixOS config flake are.

    PREFER this over reading flake.lock yourself. Reports each direct input's pin
    AGE in days. It does NOT claim an input is out of date: proving a newer commit
    exists needs a network fetch this read-only tool does not perform, so report
    ages rather than asserting anything is behind.
    """
    from .host import render

    return render.render_flake_status(_inspector().flake_status())


# --- host actions (propose only) ---------------------------------------------
#
# An agent may ASK for a privileged host change. It cannot make one.
#
# There is deliberately no approve tool here, and there never will be: an
# approval is an operator act, gated on a real session by the middleware
# (auth.OPERATOR_ONLY_PATTERN). The absence is enforced twice - by there being
# no tool, and by the HTTP endpoint refusing the machine bearer token these
# subprocesses hold. `tests/test_mcp_server.py` asserts the absence, so a future
# convenience tool cannot quietly appear.


@mcp.tool()
def propose_host_action(action: str, unit: str = "", days: int = 0) -> str:
    """Propose a privileged change to THIS host, for the operator to approve.

    Nothing happens when you call this. It returns a PREVIEW - what would
    change, what else it reaches, and how it could be undone - and leaves the
    action waiting for the operator. You cannot approve it; only a human with a
    dashboard session can.

    Use it for "restart that service" and "clean up disk space" instead of
    trying to run systemctl or nix-collect-garbage in the shell, which will fail:
    those need root, and the only route to root on this box is this proposal.

    `action` is one of: unit_start, unit_stop, unit_restart, unit_reload (pass
    `unit`, e.g. "nginx" or "nginx.service"), gc_store (no arguments), or
    gc_older_than (pass `days`).

    Show the operator the preview text verbatim rather than summarising it - the
    label saying whether it is a simulation or a statement of current state is
    part of the answer, not decoration.
    """
    args: dict[str, object] = {}
    if unit:
        args["unit"] = unit
    if days:
        args["days"] = days
    # Name ourselves in the audit. The API derives the ACTOR from the credential
    # (this subprocess presents the machine token, so it is recorded as an agent
    # whatever it claims here); this only says WHICH agent, so a record names
    # something more useful than "an agent" (review round 1, R1.6).
    answer = _api_call(
        "POST",
        "/api/host/actions",
        body={
            "kind": action,
            "args": args,
            "agent": os.environ.get("SCUFRIS_AGENT_ID", "orchestrator"),
        },
    )
    return _render_host_action(answer)


def _render_host_action(answer: str) -> str:
    """Render a host action response as the operator-facing text.

    The tool asks the model to show the preview verbatim, so it hands it prose
    rather than JSON to paraphrase - the label saying whether this is a
    simulation or a statement of current state is part of the answer, and a model
    summarising a JSON blob is exactly where that gets dropped (review round 1,
    R1.11).

    A non-JSON answer is an `error: ...` line from `_api_call`; pass it through
    unchanged rather than turning a diagnosable failure into a parse error.
    """
    from .host_actions import HostActionRecord, render_action

    try:
        payload = json.loads(answer)
    except ValueError:
        return answer
    try:
        record = HostActionRecord.model_validate(payload)
    except Exception:  # noqa: BLE001 - an unexpected shape is still an answer
        return answer
    return (
        f"{render_action(record)}\n\n"
        "This is a PROPOSAL. Nothing has happened yet, and you cannot approve "
        "it - the operator must, in the dashboard. Show them the preview above "
        "as it is written."
    )


@mcp.tool()
def host_action_status(action_id: str = "") -> str:
    """What has happened to a proposed host action (or all of them).

    Use it after `propose_host_action` to tell the operator whether their
    approval has landed and what the result was. With no id, lists the queue.
    """
    path = f"/api/host/actions/{action_id}" if action_id else "/api/host/actions"
    return _api_call("GET", path)


@mcp.tool()
def host_action_audit(limit: int = 20) -> str:
    """The record of privileged host actions: requested, refused, approved, applied.

    Written by the root helper itself, so it is the authoritative answer to
    "what has been done to this box", including actions this agent never saw.
    """
    return _api_call("GET", f"/api/host/audit?limit={max(1, min(500, limit))}")


# --- orchestrator observation (read-only) -----------------------------------
#
# These let the MAIN chat agent (the orchestrator) see the agents running on
# projects, so it can answer "what is agent-N working on". They run in THIS MCP
# subprocess, which does not share the dashboard's in-memory Supervisor - so they
# read PERSISTED state: the AgentStore (agents.json, whose lifecycle the run
# engine persists via mark_running/mark_finished) plus the backend's read_status
# from the rollout/session files. Read-only: no launching or steering (v1).


def _agent_store(settings: "Settings") -> "AgentStore":
    from .agent_store import AgentStore
    from .projects import ProjectStore

    return AgentStore(settings, ProjectStore(settings))


def _list_agents_text(settings: "Settings") -> str:
    # The list always contains at least the reserved orchestrator, so there is
    # no empty case.
    agents = _agent_store(settings).list()
    header = f"{'ID':<20} {'STATE':<9} {'BACKEND':<10} {'PROJECT':<16} NAME"
    lines = [header]
    for a in agents:
        lines.append(
            f"{a.id[:20]:<20} {a.state[:9]:<9} {a.backend[:10]:<10} "
            f"{a.project_id[:16]:<16} {a.name}"
        )
    return "\n".join(lines)


def _agent_status_text(settings: "Settings", agent_id: str) -> str:
    from .agent_store import AgentNotFound, AgentState
    from .backends import get_backend

    store = _agent_store(settings)
    try:
        agent = store.get(agent_id)
    except AgentNotFound:
        return f"error: no such agent: {agent_id}"
    lines = [
        f"agent {agent.id} ({agent.name})",
        f"state: {agent.state}",
        f"backend: {agent.backend}",
        f"project: {agent.project_id}",
        f"goal: {agent.goal or '-'}",
        f"mode: {agent.permission_mode}",
    ]
    # A run that ended in error carries WHY on its durable outcome message (a
    # backend StreamError's detail: idle timeout, over-limit line, thread-setup
    # failure). read_status below only reports session progress, never this, so
    # surface the outcome's error explicitly instead of leaving "state: error"
    # with no reason. The outcome is the cross-process substitute for the closed
    # run bus, so this works from the MCP subprocess.
    outcome = store.outcome(agent_id)
    if outcome is not None and outcome.state == AgentState.ERROR and outcome.message:
        # Flatten + cap like the pending_agents row (mcp_server.py pending loop):
        # a StreamError detail can be long or multi-line, and this line sits in a
        # single-line-per-field status block.
        detail = outcome.message.replace("\n", " ")[:200]
        lines.append(f"error: {detail}")
    if outcome is not None and outcome.state == AgentState.CANCELLED:
        # A user stop is a neutral terminal state (not an error); surface it so
        # the orchestrator does not re-read a stale prior message as if live.
        lines.append("cancelled: the run was stopped")
    try:
        status = get_backend(agent.backend).read_status(settings, agent.session_id)
    except Exception as exc:  # noqa: BLE001 - never fail the tool on a read
        status = None
        lines.append(f"(progress unavailable: {exc})")
    if status is not None:
        lines += [
            f"turns: {status.turns}",
            f"tool calls: {status.tool_calls}",
            f"tokens in/out: {status.input_tokens}/{status.output_tokens}",
            f"last message: {status.last_message or '-'}",
        ]
    return "\n".join(lines)


@mcp.tool()
def list_agents() -> str:
    """List the orchestrator's configured agents and their current state, so you
    can answer "what agents exist and what is each doing".

    Read-only. One row per agent: id, state (idle/running/blocked/done/error),
    backend (codex/claude), project, and name. Use `agent_status(id)` for detail.
    """
    from .config import Settings

    return _list_agents_text(Settings())


@mcp.tool()
def agent_status(agent_id: str) -> str:
    """Report one agent's current state and progress, so you can answer "what is
    agent-<id> working on".

    Read-only. Returns the agent's config (backend, project, goal, write posture),
    its lifecycle state, and - from its session - turns, tool calls, token usage
    and the last message. Returns a clear error if the id is unknown. This
    OBSERVES an agent; it does not launch or steer it.
    """
    from .config import Settings

    return _agent_status_text(Settings(), agent_id)


# --- orchestrator control (write) -------------------------------------------
#
# These let the orchestrator DO the dashboard's control actions - create/run/
# steer agents, create/list projects - by calling the dashboard's OWN HTTP API
# at 127.0.0.1:<port>. The MCP server is a separate process and cannot touch the
# live in-app Supervisor (the read-only observe tools above work around that by
# reading persisted state); a control action that launches or steers a run needs
# the app process that owns the supervisor, so it crosses back over HTTP, reusing
# every endpoint's validation and lifecycle. The base URL is injected as
# ``SCUFRIS_API_BASE`` when the dashboard spawns this (orchestrator-only) server.
# These tools are only registered for the orchestrator (see agent._mcp_overrides),
# so a regular agent can never create agents or projects.


def _clean_id(value: str) -> str | None:
    """An id argument, stripped; None if empty or holding a `/` or whitespace that
    would break the URL segment it is interpolated into (makes the boundary
    explicit instead of relying on the server to 404)."""
    value = value.strip()
    if not value or "/" in value or any(c.isspace() for c in value):
        return None
    return value


def _provided(**fields: object | None) -> dict[str, object]:
    """A PATCH body of only the fields the caller actually set (not None).

    ``ProjectUpdate`` / ``AgentUpdate`` are ``extra="forbid"`` and all-optional, so
    an unset field must be OMITTED, not sent as null. (An explicit empty string is
    kept - that legitimately clears a field.)"""
    return {name: value for name, value in fields.items() if value is not None}


def _reject_orchestrator(agent_id: str) -> str | None:
    """An ``error:`` message if ``agent_id`` is the reserved orchestrator, else None.

    The orchestrator configures itself through the settings store, not these tools,
    and cannot be deleted; these write tools edit REGULAR agents only."""
    from .agent_store import ORCHESTRATOR_ID

    if agent_id == ORCHESTRATOR_ID:
        return (
            f"error: {ORCHESTRATOR_ID!r} configures itself via settings and cannot "
            "be edited or deleted with this tool"
        )
    return None


@mcp.tool()
def list_projects() -> str:
    """List the registered projects (id, name, language, path), so you can pick one
    to create an agent on or answer "what projects exist".

    Read-only. One row per project."""
    text = _api_call("GET", "/api/projects")
    if text.startswith("error:"):
        return text
    try:
        projects = json.loads(text)
    except ValueError:
        return text
    if not projects:
        return "no projects registered"
    header = f"{'ID':<20} {'LANGUAGE':<12} {'NAME':<20} PATH"
    lines = [header]
    for p in projects:
        lines.append(
            f"{str(p.get('id', ''))[:20]:<20} {str(p.get('language', ''))[:12]:<12} "
            f"{str(p.get('name', ''))[:20]:<20} {p.get('cwd', '')}"
        )
    return "\n".join(lines)


@mcp.tool()
def create_project(
    name: str, cwd: str, language: str = "", description: str = ""
) -> str:
    """Register an EXISTING directory as a project (so agents can be created on it).

    ``cwd`` is the absolute path of the project directory. Returns the created
    project, or a clear error (422 bad name/path, 409 duplicate, 403 read-only)."""
    name = name.strip()
    cwd = cwd.strip()
    if not name or not cwd:
        return "error: name and cwd are required"
    return _api_call(
        "POST",
        "/api/projects",
        body={
            "name": name,
            "cwd": cwd,
            "language": language,
            "description": description,
        },
    )


@mcp.tool()
def get_project(project_id: str) -> str:
    """Show one project's detail (id, name, language, path, description) by id.

    Read-only. 404 if the id is unknown."""
    pid = _clean_id(project_id)
    if pid is None:
        return "error: project_id is required (no '/' or whitespace)"
    return _api_call("GET", f"/api/projects/{pid}")


@mcp.tool()
def update_project(
    project_id: str,
    name: str | None = None,
    cwd: str | None = None,
    language: str | None = None,
    description: str | None = None,
) -> str:
    """Edit a registered project - only the fields you pass change.

    Returns the updated project. 404 unknown, 422 invalid, 403 read-only."""
    pid = _clean_id(project_id)
    if pid is None:
        return "error: project_id is required (no '/' or whitespace)"
    body = _provided(name=name, cwd=cwd, language=language, description=description)
    if not body:
        return "error: nothing to update - pass at least one field to change"
    return _api_call("PATCH", f"/api/projects/{pid}", body=body)


@mcp.tool()
def delete_project(project_id: str) -> str:
    """Delete (unregister) a project by id.

    Removes the project record; it does not delete the directory on disk. 404
    unknown, 403 read-only."""
    pid = _clean_id(project_id)
    if pid is None:
        return "error: project_id is required (no '/' or whitespace)"
    return _api_call("DELETE", f"/api/projects/{pid}")


@mcp.tool()
def create_agent(
    name: str,
    project_id: str,
    backend: str | None = None,
    model: str | None = None,
    description: str = "",
    goal: str = "",
    permission_mode: str = "manual",
) -> str:
    """Create an agent bound to a project (from ``list_projects``).

    ``backend`` is codex or claude (server default when omitted); ``permission_mode``
    is manual|edit|auto (read-only by default). Returns the created agent, or a clear
    error (422 bad field / unknown project, 403 read-only)."""
    name = name.strip()
    project_id = project_id.strip()
    if not name or not project_id:
        return "error: name and project_id are required"
    body: dict[str, object] = {
        "name": name,
        "project_id": project_id,
        "description": description,
        "goal": goal,
        "permission_mode": permission_mode,
    }
    if backend:
        body["backend"] = backend
    if model:
        body["model"] = model
    return _api_call("POST", "/api/agents", body=body)


@mcp.tool()
def run_agent(agent_id: str, goal: str | None = None) -> str:
    """Launch a supervised background run for an agent, on its project.

    ``goal`` overrides the agent's stored goal for this run (required if the agent
    has none). Returns immediately with the run's state (usually "queued"); use
    ``agent_status`` to follow progress. 404 unknown agent, 422 no goal, 409 a run
    is already active."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    body: dict[str, str] = {}
    if goal is not None:
        body["goal"] = goal
    parent = _orch_session_id()
    if parent:
        body["parent_session_id"] = parent
    return _api_call("POST", f"/api/agents/{aid}/run", body=body)


@mcp.tool()
def message_agent(agent_id: str, message: str) -> str:
    """Send one chat turn to an agent (steer it / ask it something), resuming its
    session, and return its reply.

    This runs a full agent turn, so it can take a while. 404 unknown agent, 422
    empty message, 409 a turn is already active."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    if not message.strip():
        return "error: message must not be empty"
    body: dict[str, str] = {"message": message}
    parent = _orch_session_id()
    if parent:
        body["parent_session_id"] = parent
    raw = _api_call(
        "POST",
        f"/api/agents/{aid}/chat",
        body=body,
        read_unbounded=True,
    )
    if raw.startswith("error:"):
        return raw
    # The chat endpoint streams SSE frames; collect the assistant reply from them.
    reply = ""
    deltas: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[len("data:") :].strip())
        except ValueError:
            continue
        kind = event.get("kind")
        if kind == "done":
            reply = (event.get("reply") or {}).get("text", "") or reply
        elif kind == "text_delta":
            deltas.append(event.get("delta", ""))
        elif kind == "error":
            return f"error: {event.get('detail', 'turn failed')}"
    text = reply or "".join(deltas)
    return (text or "(no reply)")[:_MAX_OUTPUT]


@mcp.tool()
def cancel_agent(agent_id: str) -> str:
    """Cancel a sub-agent's in-flight run (stop what it is currently doing).

    Use this when asked to "cancel that agent" / "stop agent-<id>". It truly
    aborts the running turn (the backend process is stopped), and the agent's
    terminal state becomes "cancelled". 404 unknown agent or no active run (there
    is nothing to cancel). You cannot cancel the orchestrator's OWN run from here
    - the user stops that with the chat stop button."""
    from .agent_store import ORCHESTRATOR_ID

    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    if aid == ORCHESTRATOR_ID:
        return (
            "error: cannot cancel your own (orchestrator) run from within it; "
            "the user stops it with the chat stop button"
        )
    return _api_call("POST", f"/api/agents/{aid}/cancel")


@mcp.tool()
def update_agent(
    agent_id: str,
    name: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    description: str | None = None,
    goal: str | None = None,
    permission_mode: str | None = None,
) -> str:
    """Edit a REGULAR agent's config - only the fields you pass change.

    ``permission_mode`` is manual|edit|auto (read-only|edit|full); ``backend`` is
    codex or claude (the provider); ``model`` is the LLM. The reserved orchestrator
    configures itself via settings and cannot be edited here. Returns the updated
    agent. 404 unknown, 422 invalid field, 403 read-only."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    rejected = _reject_orchestrator(aid)
    if rejected is not None:
        return rejected
    body = _provided(
        name=name,
        backend=backend,
        model=model,
        description=description,
        goal=goal,
        permission_mode=permission_mode,
    )
    if not body:
        return "error: nothing to update - pass at least one field to change"
    return _api_call("PATCH", f"/api/agents/{aid}", body=body)


@mcp.tool()
def delete_agent(agent_id: str) -> str:
    """Delete a REGULAR agent by id.

    The reserved orchestrator cannot be deleted. 404 unknown, 403 read-only or
    reserved."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    rejected = _reject_orchestrator(aid)
    if rejected is not None:
        return rejected
    return _api_call("DELETE", f"/api/agents/{aid}")


@mcp.tool()
def pending_agents() -> str:
    """List the sub-agents that need YOU: those that called `request_input` and are
    waiting for a decision, those that called `report_back` and have finished, or
    those that errored. Poll this - especially at the end of a turn - so a stalled or
    finished sub-agent does not go unnoticed.

    Read-only. One row per pending agent: id, state (waiting/reported/error) and its
    question / result summary / last message. Scoped to THIS chat: children this chat
    spawned, plus unattributed ones (UI-launched), but not another chat's children
    (part 3). A `waiting` agent you answer by resuming it (`message_agent`); a
    `reported` agent has finished, so just read its report. Then call
    `acknowledge(id)` so it stops showing here."""
    parent = _orch_session_id()
    path = "/api/agents/pending"
    if parent:
        path = f"{path}?parent_session_id={quote(parent, safe='')}"
    text = _api_call("GET", path)
    if text.startswith("error:"):
        return text
    try:
        rows = json.loads(text)
    except ValueError:
        return text
    if not rows:
        return "no agents are waiting for you"
    header = f"{'ID':<20} {'STATE':<8} {'PARENT':<12} MESSAGE"
    lines = [header]
    for r in rows:
        msg = str(r.get("message", "")).replace("\n", " ")[:120]
        # Which chat spawned this child ("-" when unattributed), so the operator
        # sees the attribution the routing is based on (part 3).
        parent_sess = str(r.get("parent_session_id") or "-")[:12]
        lines.append(
            f"{str(r.get('agent_id', ''))[:20]:<20} "
            f"{str(r.get('state', ''))[:8]:<8} "
            f"{parent_sess:<12} {msg}"
        )
    return "\n".join(lines)


@mcp.tool()
def acknowledge(agent_id: str) -> str:
    """Mark an agent's pending signal handled, so it stops showing in
    `pending_agents()`. Call after you have answered its `request_input` question,
    read its `report_back` result, or dealt with its error. Idempotent - acking an
    agent with nothing pending is a harmless no-op."""
    aid = _clean_id(agent_id)
    if aid is None:
        return "error: agent_id is required (no '/' or whitespace)"
    return _api_call("POST", f"/api/agents/{aid}/acknowledge")


def _orch_session_id() -> str:
    """The orchestrator's current chat (``SCUFRIS_ORCH_SESSION_ID``, injected on the
    orchestrator MCP server per turn), so ``message_agent`` / ``run_agent`` can
    stamp a spawned child with the chat that launched it and ``pending_agents`` can
    scope to that chat (part 3). Empty on a fresh orchestrator turn."""
    import os

    return os.environ.get("SCUFRIS_ORCH_SESSION_ID", "").strip()


def main() -> None:
    """Run the orchestrator agentic MCP server over stdio (spawned by a backend).

    This is a separate process from the dashboard, so it configures its own
    logging from ``SCUFRIS_LOG_LEVEL`` (to stderr; the backend captures it). Only
    an orchestrator turn registers this server, so there is no role filtering to
    do - the audience split is physical (see the module docstring); the operator
    disabled-tool set is still applied here.
    """
    import os

    from .logsetup import configure_logging

    configure_logging(os.environ.get("SCUFRIS_LOG_LEVEL", "INFO"))
    removed = apply_disabled_tools(mcp, _disabled_tools())
    if removed:
        logger.info("disabled tools: %s", ", ".join(removed))
    mcp.run()


if __name__ == "__main__":
    main()
