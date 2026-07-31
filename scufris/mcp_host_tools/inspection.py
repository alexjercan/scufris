"""The read-only inspection tools: units, journal, storage, network, thermals.

The deep read-only surface over this NixOS box. Everything runs through
`scufris.host`, so each tool is bounded, classified and honest about what it
could not read - see that package's docstring. Nothing here mutates the system;
the mutating half is the separate, approval-gated surface in `actions`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..mcp_common import _run
from ..metrics import PsutilCollector
from ..processes import ProcessList, PsutilProcessCollector

if TYPE_CHECKING:
    # Imported lazily inside the tool helpers (to keep an MCP server's startup
    # import light); named here only for type checking.
    from ..host import HostInspector, Scope

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


def _inspector() -> "HostInspector":
    """A HostInspector configured from settings.

    Built per call rather than at import: this module is imported by the
    dashboard as well as run as an MCP subprocess, and `Settings()` at import
    time would freeze whatever the environment looked like then.
    """
    from ..config import Settings
    from ..host import HostInspector

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
    from ..host import Scope

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
    from ..host import render

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
    from ..host import render

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
    from ..host import render

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
    from ..host import render

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
    from ..host import render

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
    from ..host import render

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
    from ..host import render

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
    from ..host import render

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
    from ..host import render

    return render.render_thermal(_inspector().thermal())


def host_what_provides(binary: str) -> str:
    """Which Nix package provides a command on this host.

    PREFER this over shell `which` + `readlink` for "where does X come from" or
    "what package is X in" - it follows the symlink chain into the store and
    reports the package name and version.
    """
    from ..host import render

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
    from ..host import render

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
    from ..host import render

    return render.render_flake_status(_inspector().flake_status())
