"""Network inspection: interfaces, who is listening, and what the firewall allows.

Two honesty constraints shape this module, both measured on this host rather
than assumed:

1. **Socket ownership is partial without privilege.** psutil sees 18 listeners
   here and can name the owning process for 7 - the rest belong to other users.
   Those rows say "owner not visible without privilege"; they are never blank,
   because a blank owner column reads as "nothing owns this".

2. **The LIVE firewall is not readable as the operator.** ``iptables -L`` fails
   with "Permission denied" here. What IS readable is the DECLARED rule set:
   NixOS builds a ``firewall-start`` script into ``/run/current-system`` listing
   every accepted port. This module reports that, labelled declared, and says the
   live table needs privilege. Presenting the declared set as "current" would be
   the same failure the host spike rejected for previews.
"""

from __future__ import annotations

import json
import re
import socket
from pathlib import Path

import psutil
from pydantic import BaseModel, Field

from .models import Availability, Bounded, Report, clamp
from .run import DEFAULT_TIMEOUT, Runner

DEFAULT_SOCKET_LIMIT = 40
MAX_SOCKET_LIMIT = 200

# The activated system's firewall script, built by the NixOS firewall module.
CURRENT_SYSTEM = Path("/run/current-system")

# `ip46tables -A nixos-fw -p tcp --dport 22 -j nixos-fw-accept`, and the range
# form `--dport 27031:27035`. Interface-scoped rules carry `-i <iface>`.
_FW_RULE = re.compile(
    r"ip(?P<family>46|6)?tables\s+-A\s+nixos-fw\b(?P<body>[^\n]*?)"
    r"-j\s+nixos-fw-(?P<verdict>accept|refuse|log-refuse)",
)
_FW_PROTO = re.compile(r"-p\s+(?P<proto>tcp|udp)\b")
_FW_PORT = re.compile(r"--dport\s+(?P<port>[\d:]+)")
_FW_IFACE = re.compile(r"-i\s+(?P<iface>[\w.@-]+)")


class InterfaceAddress(BaseModel):
    family: str  # inet | inet6
    address: str
    prefix_length: int = 0
    scope: str = ""


class NetworkInterface(BaseModel):
    name: str
    state: str = ""
    mac: str = ""
    mtu: int = 0
    addresses: list[InterfaceAddress] = Field(default_factory=list)

    @property
    def up(self) -> bool:
        return self.state.upper() in {"UP", "UNKNOWN"}


class ListeningSocket(BaseModel):
    """One listening socket, with the owner when it is visible."""

    protocol: str
    address: str
    port: int
    pid: int | None = None
    process: str = ""
    # True when the socket belongs to another user, so psutil cannot name it.
    owner_hidden: bool = False

    @property
    def exposed(self) -> bool:
        """Bound to something other than loopback - reachable off this machine."""
        return not (
            self.address.startswith("127.") or self.address in {"::1", "localhost"}
        )


class FirewallRule(BaseModel):
    protocol: str = ""
    ports: str = ""
    interface: str = ""
    verdict: str = "accept"


class FirewallReport(Report):
    """The DECLARED firewall, never the live table. See the module docstring."""

    source: str = ""
    rules: list[FirewallRule] = Field(default_factory=list)
    # Always set: this report is structurally incapable of showing the live
    # ruleset, and the caller must never render it as if it did.
    declared_only: bool = True

    def _allowed(self, protocol: str) -> list[str]:
        """Ports open for ``protocol``, deduplicated in declaration order.

        NixOS emits the same port more than once when it is opened both globally
        and per-interface (11433 appears twice on this host), and a summary line
        that repeats a port reads as two different openings.
        """
        seen: dict[str, None] = {}
        for rule in self.rules:
            if rule.protocol == protocol and rule.ports:
                seen.setdefault(rule.ports, None)
        return list(seen)

    @property
    def allowed_tcp(self) -> list[str]:
        return self._allowed("tcp")

    @property
    def allowed_udp(self) -> list[str]:
        return self._allowed("udp")


class InterfaceReport(Report):
    interfaces: list[NetworkInterface] = Field(default_factory=list)


class SocketReport(Report, Bounded):
    sockets: list[ListeningSocket] = Field(default_factory=list)

    @property
    def hidden_owners(self) -> int:
        return sum(1 for s in self.sockets if s.owner_hidden)


class NetworkReport(Report):
    interfaces: InterfaceReport = Field(default_factory=InterfaceReport)
    listening: SocketReport = Field(default_factory=SocketReport)
    firewall: FirewallReport = Field(default_factory=FirewallReport)


def list_interfaces(
    runner: Runner, *, timeout: float = DEFAULT_TIMEOUT
) -> InterfaceReport:
    """Interfaces and addresses from ``ip -j addr`` (machine-readable)."""
    result = runner(["ip", "-j", "addr"], timeout=timeout)
    report = InterfaceReport(available=Availability.of(result))
    if not result.ok:
        return report
    try:
        rows = json.loads(result.stdout or "[]")
    except ValueError as exc:
        report.available = Availability.unavailable(
            f"ip returned output this build cannot parse: {exc}"
        )
        return report
    if not isinstance(rows, list):
        report.available = Availability.unavailable(
            "ip returned a non-list where an interface array was expected"
        )
        return report
    for row in rows:
        if not isinstance(row, dict):
            continue
        addresses = [
            InterfaceAddress(
                family=str(addr.get("family", "")),
                address=str(addr.get("local", "")),
                prefix_length=int(addr.get("prefixlen", 0) or 0),
                scope=str(addr.get("scope", "")),
            )
            for addr in row.get("addr_info", [])
            if isinstance(addr, dict) and addr.get("local")
        ]
        report.interfaces.append(
            NetworkInterface(
                name=str(row.get("ifname", "")),
                state=str(row.get("operstate", "")),
                mac=str(row.get("address", "")),
                mtu=int(row.get("mtu", 0) or 0),
                addresses=addresses,
            )
        )
    return report


def listening_sockets(*, limit: int = DEFAULT_SOCKET_LIMIT) -> SocketReport:
    """Listening TCP/UDP sockets, with owners where privilege allows.

    psutil rather than ``ss``: same visibility, no subprocess, and the pid comes
    back as a number instead of a string to re-parse.

    Bounded like every other listing: this host produced 54 rows on a live run,
    most of them netbios broadcast sockets, so an uncapped list is a context
    budget spent on noise.
    """
    capped = clamp(limit, default=DEFAULT_SOCKET_LIMIT, maximum=MAX_SOCKET_LIMIT)
    report = SocketReport(limit=capped)
    rows: list[ListeningSocket] = []
    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, PermissionError) as exc:
        report.available = Availability.unavailable(
            f"the socket table could not be read: {exc}"
        )
        return report
    except (OSError, RuntimeError) as exc:
        report.available = Availability.unavailable(
            f"the socket table could not be read: {exc}"
        )
        return report
    for conn in connections:
        is_tcp = conn.type == socket.SOCK_STREAM
        # UDP has no LISTEN state; a bound UDP socket with no peer is the
        # equivalent "something is waiting here".
        if is_tcp and conn.status != psutil.CONN_LISTEN:
            continue
        if not is_tcp and conn.raddr:
            continue
        # psutil types laddr as `addr | tuple[()]` - an unbound socket carries the
        # empty tuple, so the fields are read defensively rather than indexed.
        address = getattr(conn.laddr, "ip", "")
        port = getattr(conn.laddr, "port", 0)
        if not address and not port:
            continue
        name = ""
        if conn.pid:
            try:
                name = psutil.Process(conn.pid).name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                name = ""
        rows.append(
            ListeningSocket(
                protocol="tcp" if is_tcp else "udp",
                address=str(address),
                port=int(port),
                pid=conn.pid,
                process=name,
                owner_hidden=conn.pid is None,
            )
        )
    rows.sort(key=lambda s: (s.protocol, s.port))
    report.total_seen = len(rows)
    report.sockets = rows[:capped]
    report.truncated = len(rows) > capped
    hidden = sum(1 for s in rows if s.owner_hidden)
    if hidden:
        report.available = Availability(
            ok=True,
            caveat=(
                f"{hidden} of {len(rows)} sockets belong to another user, so their "
                "owning process is not visible without privilege"
            ),
        )
    return report


def _firewall_script(system: Path = CURRENT_SYSTEM) -> Path | None:
    """The firewall-start script the ACTIVATED system references.

    Found through the firewall unit rather than by globbing the store, so it is
    the script this generation actually runs, not some other build lying around.
    """
    unit = system / "etc" / "systemd" / "system" / "firewall.service"
    try:
        text = unit.read_text(errors="replace")
    except OSError:
        return None
    # Any absolute path ending in the firewall-start layout, not specifically a
    # /nix/store one: the shape is what identifies it, and pinning the store
    # prefix would make this untestable against a fixture tree for no gain in
    # precision (the path still has to come out of THIS system's unit file).
    for match in re.finditer(r"(/\S+?-firewall-start/bin/firewall-start)", text):
        candidate = Path(match.group(1))
        if candidate.is_file():
            return candidate
    return None


def declared_firewall(system: Path = CURRENT_SYSTEM) -> FirewallReport:
    """The ports the ACTIVATED NixOS configuration opens.

    This is the declared set, not the live table (which needs root here). The
    ``declared_only`` flag is always true and every renderer must say so.
    """
    report = FirewallReport()
    script = _firewall_script(system)
    if script is None:
        report.available = Availability.unavailable(
            "no firewall-start script was found under "
            f"{system} - either the NixOS firewall is off or this is not a "
            "NixOS system. The live rule table needs root and is not readable "
            "from here either way."
        )
        return report
    try:
        text = script.read_text(errors="replace")
    except OSError as exc:
        report.available = Availability.unavailable(
            f"the firewall script at {script} could not be read: {exc}"
        )
        return report
    report.source = str(script)
    for match in _FW_RULE.finditer(text):
        body = match.group("body")
        port = _FW_PORT.search(body)
        proto = _FW_PROTO.search(body)
        iface = _FW_IFACE.search(body)
        if port is None and proto is None:
            # Structural rules (loopback accept, conntrack, the final refuse)
            # are not "an opening" and only add noise to the answer.
            continue
        report.rules.append(
            FirewallRule(
                protocol=proto.group("proto") if proto else "",
                ports=port.group("port") if port else "",
                interface=iface.group("iface") if iface else "",
                verdict=match.group("verdict"),
            )
        )
    report.available = Availability(
        ok=True,
        caveat=(
            "these are the rules the current NixOS generation DECLARES; the live "
            "iptables table needs root and was not read"
        ),
    )
    return report


def network_report(
    runner: Runner,
    *,
    system: Path = CURRENT_SYSTEM,
    timeout: float = DEFAULT_TIMEOUT,
) -> NetworkReport:
    """Interfaces, listening sockets and the declared firewall together."""
    interfaces = list_interfaces(runner, timeout=timeout)
    listening = listening_sockets()
    firewall = declared_firewall(system)
    report = NetworkReport(
        interfaces=interfaces, listening=listening, firewall=firewall
    )
    broken = [
        part.available.reason
        for part in (interfaces, listening, firewall)
        if not part.ok
    ]
    if len(broken) == 3:
        report.available = Availability.unavailable("; ".join(broken))
    elif broken:
        report.available = Availability(ok=True, caveat="; ".join(broken))
    return report
