"""Compact text renderings of the host reports, for the agent's tool results.

The MCP tools return TEXT rather than JSON for the same reason ``list_processes``
does: a model reads a small aligned table better than a nested object, and the
budget goes to data instead of punctuation.

Every renderer here obeys the package's central rule at the point where it is
easiest to break: a report that is unavailable prints ``unavailable: <reason>``,
a report that is available but EMPTY prints an explicit statement of the absence
("no failed units"), and a report that was capped prints its truncation marker.
There is no code path that emits a blank section, because a blank section reads
to a model as "I checked and everything is fine".
"""

from __future__ import annotations

from .journal import JournalReport
from .models import Bounded, Report
from .network import FirewallReport, NetworkReport, SocketReport
from .packages import BinaryProvider, ClosureDiff, FlakeStatus, ProfileReport
from .storage import LargestDirectories, ReclaimableSpace, StorageReport
from .thermal import ThermalReport
from .units import UnitList, UnitStatus

_PRIORITY_WORDS = {
    0: "emerg",
    1: "alert",
    2: "crit",
    3: "err",
    4: "warn",
    5: "notice",
    6: "info",
    7: "debug",
}


def human_bytes(num: float) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def _header(report: Report, title: str) -> tuple[list[str], bool]:
    """``(lines, available)``. When unavailable, ``lines`` is already complete.

    Returning the availability flag alongside the lines is what stops a renderer
    from silently skipping the branch: the caller cannot get the header without
    also being handed the answer to "is there a body to render at all".
    """
    if not report.ok:
        return [title, f"unavailable: {report.available.reason}"], False
    lines = [title]
    note = report.available.line()
    if note:
        lines.append(note)
    return lines, True


def _truncation(report: Bounded, lines: list[str]) -> None:
    marker = report.truncation_marker()
    if marker:
        lines.append(marker)


def render_units(report: UnitList) -> str:
    scope = report.scope
    label = (
        f"{scope} units in state {report.state_filter!r}"
        if report.state_filter
        else f"{scope} units"
    )
    lines, available = _header(report, label)
    if not available:
        return "\n".join(lines)
    if not report.units:
        # The distinction this package exists for: asked and answered, nothing
        # there - never a blank block.
        lines.append(
            f"no {scope} units matched"
            if report.state_filter != "failed"
            else f"no failed {scope} units - nothing is in a failed state"
        )
        return "\n".join(lines)
    lines.append(f"{'UNIT':<40} {'ACTIVE':<10} {'SUB':<10} DESCRIPTION")
    for unit in report.units:
        lines.append(
            f"{unit.name[:40]:<40} {unit.active[:10]:<10} "
            f"{unit.sub[:10]:<10} {unit.description[:60]}"
        )
    _truncation(report, lines)
    return "\n".join(lines)


def render_unit_status(status: UnitStatus) -> str:
    lines, available = _header(status, f"{status.scope} unit {status.name}")
    if not available:
        return "\n".join(lines)
    if not status.loaded and status.load_state:
        # `systemctl show` answers for a unit that does not exist, so say that
        # rather than printing a status block full of empty strings.
        lines.append(f"load state: {status.load_state} - no such unit is loaded")
        return "\n".join(lines)
    lines += [
        f"description: {status.description or '-'}",
        f"state: {status.active_state}/{status.sub_state} "
        f"(unit file: {status.unit_file_state or '-'})",
        f"result of last run: {status.result or '-'}",
        f"restarts: {status.restarts}",
        f"main pid: {status.main_pid or '-'}",
        f"started: {status.started_at or '-'}",
    ]
    if status.exit_status is not None and status.active_state != "active":
        lines.append(f"last exit status: {status.exit_status}")
    if status.memory_bytes is not None and status.memory_bytes >= 0:
        lines.append(f"memory: {human_bytes(status.memory_bytes)}")
    if status.status_text:
        lines.append(f"status: {status.status_text}")
    return "\n".join(lines)


def render_journal(report: JournalReport) -> str:
    query = report.query
    target = query.unit or f"all {query.scope} units"
    window = f" since {query.since}" if query.since else ""
    lines, available = _header(report, f"journal for {target}{window}")
    if not available:
        return "\n".join(lines)
    if not report.entries:
        # The truncation check comes FIRST. A single message larger than the
        # whole byte budget empties the entry list while `truncated` is set, and
        # falling through to "the window is empty" would report "nothing
        # happened" about data that was read and then dropped - the exact
        # confusion this module exists to prevent.
        if report.truncated:
            lines.append(
                "NOT EMPTY: the matching entries were too large for this query's "
                f"byte budget, so none are shown (saw {report.total_seen} raw "
                "lines). Narrow the unit or the time window."
            )
        elif report.unparsed:
            lines.append(
                f"NOT EMPTY: {report.unparsed} entries matched but none could be "
                "parsed by this build, so none are shown"
            )
        else:
            lines.append(
                "no journal entries matched this query - the window is empty, "
                "not broken"
            )
        return "\n".join(lines)
    for entry in report.entries:
        stamp = entry.timestamp.strftime("%m-%d %H:%M:%S") if entry.timestamp else "-"
        level = _PRIORITY_WORDS.get(entry.priority, str(entry.priority))
        unit = f" {entry.unit}" if entry.unit and not query.unit else ""
        lines.append(f"{stamp} {level:<6}{unit} {entry.message}")
    if report.severe_count:
        lines.append(f"({report.severe_count} of these are err or worse)")
    if report.unparsed:
        lines.append(f"({report.unparsed} entries could not be parsed and are missing)")
    if report.bytes_truncated:
        lines.append(
            f"... cut at the {report.limit}-line window's byte budget - "
            "narrow the unit or time range"
        )
    else:
        _truncation(report, lines)
    return "\n".join(lines)


def render_storage(report: StorageReport) -> str:
    lines, available = _header(report, "storage")
    if not available:
        return "\n".join(lines)
    filesystems = report.filesystems
    if not filesystems.ok:
        lines.append(f"filesystems unavailable: {filesystems.available.reason}")
    elif not filesystems.filesystems:
        lines.append("no real filesystems were found (only pseudo-filesystems)")
    else:
        lines.append(f"{'MOUNT':<24} {'USED':>9} {'TOTAL':>9} {'FREE':>9}  USE%")
        for fs in filesystems.filesystems:
            lines.append(
                f"{fs.mountpoint[:24]:<24} {human_bytes(fs.used):>9} "
                f"{human_bytes(fs.total):>9} {human_bytes(fs.free):>9} "
                f"{fs.percent:>5.1f}%"
            )
    if report.nix_store is not None:
        store = report.nix_store
        lines.append(
            f"nix store lives on {store.mountpoint} "
            f"({human_bytes(store.free)} free of {human_bytes(store.total)})"
        )
    generations = report.generations
    if not generations.ok:
        lines.append(f"generations unavailable: {generations.available.reason}")
    elif not generations.generations:
        lines.append("no NixOS system generations were listed")
    else:
        current = generations.current
        lines.append(
            f"generations: {len(generations.generations)} "
            f"(current: {current.number if current else 'unknown'})"
        )
        for gen in generations.generations[:5]:
            marker = " <- current" if gen.current else ""
            lines.append(
                f"  {gen.number:<5} {gen.date:<20} kernel {gen.kernel_version}{marker}"
            )
    return "\n".join(lines)


def render_largest_directories(report: LargestDirectories) -> str:
    lines, available = _header(
        report, f"largest directories under {report.root} (depth {report.depth})"
    )
    if not available:
        return "\n".join(lines)
    if not report.directories:
        lines.append(f"no subdirectories were found under {report.root}")
        return "\n".join(lines)
    for entry in report.directories:
        lines.append(f"{human_bytes(entry.bytes):>9}  {entry.path}")
    _truncation(report, lines)
    return "\n".join(lines)


def render_reclaimable(report: ReclaimableSpace) -> str:
    lines, available = _header(report, "garbage-collectable store paths")
    if not available:
        return "\n".join(lines)
    if report.dead_paths is None:
        lines.append("no dead-path count was reported")
        return "\n".join(lines)
    if report.dead_paths == 0:
        lines.append("nothing would be collected - the store has no dead paths")
        return "\n".join(lines)
    lines.append(f"{report.dead_paths} store paths are dead and could be deleted")
    # Stated rather than silently omitted: nix reports a count and no size, and
    # dressing the count up as a size is the failure the host spike rejected.
    lines.append(
        "nix reports a path COUNT and no byte total, so this is not a size. "
        "Collecting them is a privileged action and is not available here."
    )
    return "\n".join(lines)


def render_sockets(report: SocketReport) -> list[str]:
    lines, available = _header(report, "listening sockets")
    if not available:
        return lines
    if not report.sockets:
        lines.append("nothing is listening on this host")
        return lines
    lines.append(f"{'PROTO':<6} {'ADDRESS':<24} {'PORT':>6}  OWNER")
    for sock in report.sockets:
        if sock.owner_hidden:
            owner = "(owner not visible without privilege)"
        elif sock.process:
            owner = f"{sock.process} (pid {sock.pid})"
        else:
            owner = f"pid {sock.pid}"
        exposed = " [exposed]" if sock.exposed else ""
        lines.append(
            f"{sock.protocol:<6} {sock.address[:24]:<24} {sock.port:>6}  "
            f"{owner}{exposed}"
        )
    _truncation(report, lines)
    return lines


def render_firewall(report: FirewallReport) -> list[str]:
    lines, available = _header(report, "firewall (DECLARED by the current generation)")
    if not available:
        return lines
    if not report.rules:
        lines.append(
            "the current generation declares no port openings - only the "
            "structural rules (loopback, established connections, final refuse)"
        )
        return lines
    tcp = ", ".join(report.allowed_tcp) or "none"
    udp = ", ".join(report.allowed_udp) or "none"
    lines.append(f"tcp open: {tcp}")
    lines.append(f"udp open: {udp}")
    scoped = [r for r in report.rules if r.interface]
    for rule in scoped:
        lines.append(f"  on {rule.interface} only: {rule.protocol} {rule.ports}")
    return lines


def render_network(report: NetworkReport) -> str:
    lines, available = _header(report, "network")
    if not available:
        return "\n".join(lines)
    interfaces = report.interfaces
    if not interfaces.ok:
        lines.append(f"interfaces unavailable: {interfaces.available.reason}")
    elif not interfaces.interfaces:
        lines.append("no network interfaces were reported")
    else:
        for iface in interfaces.interfaces:
            addresses = (
                ", ".join(f"{a.address}/{a.prefix_length}" for a in iface.addresses)
                or "no address"
            )
            lines.append(f"{iface.name:<12} {iface.state:<8} {addresses}")
    lines.append("")
    lines += render_sockets(report.listening)
    lines.append("")
    lines += render_firewall(report.firewall)
    return "\n".join(lines)


def render_thermal(report: ThermalReport) -> str:
    lines, available = _header(report, "thermal and power")
    if not available:
        return "\n".join(lines)
    if not report.temperatures:
        lines.append("no temperature sensors were readable")
    else:
        for reading in report.temperatures[:12]:
            limit = f" (limit {reading.limit:.0f}C)" if reading.limit else ""
            flag = " NEAR LIMIT" if reading.near_limit else ""
            lines.append(
                f"{reading.chip}/{reading.label}: {reading.celsius:.1f}C{limit}{flag}"
            )
    throttle = report.throttling
    if not throttle.ok:
        lines.append(f"throttling unknown: {throttle.available.reason}")
    elif throttle.throttled:
        # The finding a temperature gauge cannot show: the CPU was held back.
        lines.append(
            f"THROTTLED since boot: {throttle.core_events} core events "
            f"({throttle.core_time_ms}ms), {throttle.package_events} package "
            f"events ({throttle.package_time_ms}ms) across "
            f"{throttle.cpus_read} cpus"
        )
    else:
        lines.append(
            f"no thermal throttling recorded since boot (across "
            f"{throttle.cpus_read} cpus)"
        )
    battery = report.battery
    if not battery.ok:
        lines.append(f"battery unknown: {battery.available.reason}")
    elif not battery.present:
        lines.append(battery.available.caveat or "no battery is present")
    else:
        plug = "on AC" if battery.plugged_in else "on battery"
        lines.append(f"battery: {battery.percent:.0f}% {plug}")
    fans = report.fans
    if not fans.ok:
        lines.append(f"fans unknown: {fans.available.reason}")
    elif not fans.present:
        lines.append(fans.available.caveat or "no fan sensors are present")
    else:
        for fan in fans.fans:
            lines.append(f"fan {fan.chip}/{fan.label}: {fan.celsius:.0f} rpm")
    return "\n".join(lines)


def render_provider(report: BinaryProvider) -> str:
    lines, available = _header(report, f"what provides {report.binary!r}")
    if not available:
        return "\n".join(lines)
    lines.append(f"path: {report.path}")
    if report.package:
        version = f" {report.version}" if report.version else ""
        lines.append(f"package: {report.package}{version}")
    lines.append(f"store path: {report.store_path or '-'}")
    return "\n".join(lines)


def render_profile(report: ProfileReport) -> str:
    lines, available = _header(report, f"profile {report.profile}")
    if not available:
        return "\n".join(lines)
    lines.append(f"{report.binary_count} binaries")
    if report.store_path:
        lines.append(f"store path: {report.store_path}")
    if report.packages:
        lines.append("first entries: " + ", ".join(report.packages))
        if report.binary_count > len(report.packages):
            lines.append(
                f"... {report.binary_count - len(report.packages)} more not shown"
            )
    return "\n".join(lines)


def render_closure_diff(report: ClosureDiff) -> str:
    lines, available = _header(
        report, f"closure diff {report.before} -> {report.after}"
    )
    if not available:
        return "\n".join(lines)
    if report.identical:
        # THE trap, defused where a reader can see it: nix prints nothing at all
        # for an identical closure, so "no change" must be said out loud.
        lines.append(
            "no closure change - the two closures are identical "
            "(nix prints nothing at all in this case, which is why it is stated "
            "here rather than shown as an empty diff)"
        )
        return "\n".join(lines)
    if not report.changes:
        lines.append(
            "nix produced output this build could not parse into changes; "
            "treat the diff as unknown rather than empty"
        )
        return "\n".join(lines)
    for change in report.changes:
        lines.append(f"{change.package}: {change.detail}")
    _truncation(report, lines)
    return "\n".join(lines)


def render_flake_status(report: FlakeStatus) -> str:
    lines, available = _header(report, f"flake inputs of {report.repository}")
    if not available:
        return "\n".join(lines)
    lines.append(f"{'INPUT':<20} {'AGE':>8}  SOURCE")
    for entry in report.inputs:
        age = entry.age_days()
        age_text = f"{age}d" if age is not None else "unknown"
        lines.append(f"{entry.name[:20]:<20} {age_text:>8}  {entry.source}")
    return "\n".join(lines)
