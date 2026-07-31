"""The host toolset: the read-only inspection tools and the propose-only actions.

Defined here ONCE and registered onto a server by ``register``, because the two
audiences that may see host tools need different halves of the same set:

- the ORCHESTRATOR (``mcp_server``, server id ``scufris``) registers the
  INSPECTION half, so "why is this box hot" stays a direct answer in the chat the
  operator is already in;
- the HOST AGENT (``host_mcp_server``, server id ``host``) registers inspection
  AND the mutating propose tools, so the propose -> preview -> approve contract is
  stated to exactly one audience and lives in one steering preamble.

The audience split is PHYSICAL: a tool reaches an audience by being registered on
a server that audience's turn wires up, never by a runtime filter. That is why
this module exports functions and a registrar instead of decorating at
definition - the same function object is registered on one server or two,
and nothing has to be filtered out afterwards.

Every tool that shells out uses a fixed argument list (never a shell string), a
timeout, and bounded output. Nothing here can CHANGE the host: the propose tools
ask the dashboard's API for a preview and leave the action waiting for an
operator, which is the only route to root on this box.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Callable

from .mcp_common import _api_call, _run
from .metrics import PsutilCollector
from .processes import ProcessList, PsutilProcessCollector

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    # Imported lazily inside the tool helpers (to keep an MCP server's startup
    # import light); named here only for type checking.
    from .host import HostInspector, Scope

logger = logging.getLogger(__name__)

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


def list_processes(limit: int = 15) -> str:
    """Top running applications by CPU, grouped by name (like a compact htop).

    PREFER this over shell `ps`/`top`/`htop` for any "what is using CPU/memory"
    question - it is already grouped and ranked for this host.
    """
    return _format_processes(_proc_collector.sample(), limit)


# --- host inspection (read-only) ---------------------------------------------
#
# The deep read-only surface over this NixOS box: units, journal, storage,
# network, thermals, packages and generations. Everything runs through
# `scufris.host`, so each tool is
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


def host_what_provides(binary: str) -> str:
    """Which Nix package provides a command on this host.

    PREFER this over shell `which` + `readlink` for "where does X come from" or
    "what package is X in" - it follows the symlink chain into the store and
    reports the package name and version.
    """
    from .host import render

    return render.render_provider(_inspector().what_provides(binary))


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


def propose_host_action(
    action: str, unit: str = "", days: int = 0, generation: int = 0
) -> str:
    """Propose a privileged change to THIS host, for the operator to approve.

    Nothing happens when you call this. It returns a PREVIEW - what would
    change, what else it reaches, and how it could be undone - and leaves the
    action waiting for the operator. You cannot approve it; only a human with a
    dashboard session can.

    Use it for "restart that service" and "clean up disk space" instead of
    trying to run systemctl or nix-collect-garbage in the shell, which will fail:
    those need root, and the only route to root on this box is this proposal.

    `action` is one of: unit_start, unit_stop, unit_restart, unit_reload (pass
    `unit`, e.g. "nginx" or "nginx.service"), gc_store (no arguments),
    gc_older_than (pass `days`), or rollback (pass `generation` - the number from
    the generation list, which returns the whole system to that configuration).

    Changing the NixOS CONFIGURATION is not here: use `propose_nixos_change`,
    which builds a commit first. `activate` is refused on this path outright,
    because what gets activated must be something the server built from an
    identified revision rather than a path handed to it.

    Show the operator the preview text verbatim rather than summarising it - the
    label saying whether it is a simulation or a statement of current state is
    part of the answer, not decoration.
    """
    args: dict[str, object] = {}
    if unit:
        args["unit"] = unit
    if days:
        args["days"] = days
    if generation:
        args["generation"] = generation
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


def host_action_status(action_id: str = "") -> str:
    """What has happened to a proposed host action (or all of them).

    Use it after `propose_host_action` to tell the operator whether their
    approval has landed and what the result was. With no id, lists the queue.
    """
    path = f"/api/host/actions/{action_id}" if action_id else "/api/host/actions"
    return _api_call("GET", path)


def propose_nixos_change(ref: str = "", repo: str = "", attr: str = "") -> str:
    """Build a COMMITTED NixOS configuration and propose activating it.

    Use this after the configuration repository has actually been changed and
    committed - which is ordinary project work, not a host action: open a
    worktree on that project, edit it, commit on a branch, review it. Then name
    that branch here.

    `ref` is a branch, tag or commit (default: HEAD of that working tree). `repo`
    is the configuration repository or one of its worktrees (default: the
    configured one). `attr` is the nixosConfiguration to build (default: this
    machine's hostname).

    What happens: the ref is resolved to a commit, that commit is BUILT as the
    operator (not as root), and if it builds, the activation is proposed for the
    operator to approve - with a closure diff against the running system as its
    preview. Nothing is activated by this call, and you cannot approve it.

    The build takes the tree from the COMMIT, so uncommitted edits are not in it.
    A build failure ends here: the log comes back and no proposal is created.
    Building can take a long time; poll `nixos_change_status`.
    """
    body: dict[str, object] = {
        "agent": os.environ.get("SCUFRIS_AGENT_ID", "orchestrator"),
    }
    if ref:
        body["ref"] = ref
    if repo:
        body["repo"] = repo
    if attr:
        body["attr"] = attr
    answer = _api_call("POST", "/api/host/config/changes", body=body)
    return _render_config_change(answer)


def nixos_change_status(change_id: str = "") -> str:
    """How a proposed NixOS configuration change is doing (or all of them).

    Use it after `propose_nixos_change`: while the state is `building` the build
    is still running, `failed` carries the build log, and `proposed` means there
    is a host action waiting for the operator - read that with
    `host_action_status` to show them the closure diff.
    """
    path = (
        f"/api/host/config/changes/{change_id}"
        if change_id
        else "/api/host/config/changes"
    )
    answer = _api_call("GET", path)
    return _render_config_change(answer) if change_id else answer


def _render_config_change(answer: str) -> str:
    """Render a config-change response as the operator-facing text.

    Same reasoning as `_render_host_action`: the notes about what is NOT in the
    build and whether the revision is merged are part of the answer, and a model
    paraphrasing JSON is exactly where they get dropped.
    """
    from .hostconfig import ConfigChange, render_change

    try:
        payload = json.loads(answer)
    except ValueError:
        return answer
    try:
        change = ConfigChange.model_validate(payload)
    except Exception:  # noqa: BLE001 - an unexpected shape is still an answer
        return answer
    text = render_change(change)
    if change.action_id:
        return (
            f"{text}\n\nThe activation is PROPOSED as host action "
            f"{change.action_id}. Read it with host_action_status and show the "
            "operator its preview verbatim; only they can approve it."
        )
    return text


def host_action_audit(limit: int = 20) -> str:
    """The record of privileged host actions: requested, refused, approved, applied.

    Written by the root helper itself, so it is the authoritative answer to
    "what has been done to this box", including actions this agent never saw.
    """
    return _api_call("GET", f"/api/host/audit?limit={max(1, min(500, limit))}")


# The two halves, named so a registration site reads as the audience it serves.
# `INSPECTION` never changes the host; `ACTIONS` can only ever ASK for a change.
INSPECTION: tuple[Callable[..., Any], ...] = (
    host_stats,
    disk_usage,
    list_processes,
    host_units,
    host_failed_units,
    host_unit_status,
    host_journal,
    host_storage,
    host_largest_directories,
    host_reclaimable_space,
    host_network,
    host_thermal,
    host_what_provides,
    host_generation_diff,
    host_flake_status,
)

# Propose-only, and there is deliberately no approve tool in this set (nor
# anywhere else): approving is an operator act gated on a real session by the
# middleware (`auth.OPERATOR_ONLY_PATTERN`), and the HTTP endpoint refuses the
# machine bearer token these subprocesses hold. `tests/test_host_mcp_server.py`
# asserts the absence, so a future convenience tool cannot quietly appear.
ACTIONS: tuple[Callable[..., Any], ...] = (
    propose_host_action,
    host_action_status,
    propose_nixos_change,
    nixos_change_status,
    host_action_audit,
)


def register(mcp: "FastMCP", *, actions: bool) -> None:
    """Register the host tools on ``mcp``: inspection always, the propose tools
    only when ``actions`` is set.

    Each function is registered under its own name and docstring, exactly as an
    ``@mcp.tool()`` decorator would - the difference is only that the audience
    decides, at registration time, which half exists on that server at all.
    """
    for tool in INSPECTION:
        mcp.tool()(tool)
    if actions:
        for tool in ACTIONS:
            mcp.tool()(tool)
