#!/usr/bin/env python
"""Render every host report from canned fixtures, touching no real host.

``host_inspect.py`` is the same tour against the live machine and needs a NixOS
box; this one needs nothing at all, which is why it is the one on
``tests/test_examples.py``'s ``OFFLINE`` list. It exists so the fourteen
renderers in ``scufris_host.render`` have an executable check that survives the
package carve: every one is called here, and a renderer that starts raising or
returning nothing fails the suite rather than a chat message.

    python examples/host_report_fixture.py

Two seams make it possible, and both are load-bearing:

* ``FakeRunner`` replays canned command output keyed by argv prefix, so every
  report backed by a subprocess (units, the journal, generations, interfaces,
  the closure diff, ``du``, the GC dry run) parses real-shaped bytes.
* ``HostInspector`` takes ``config_repo``, ``system`` and ``cpu_sysfs`` as
  paths, so every report backed by the filesystem reads a tree this script
  builds under a tempdir.

Three collectors have NO seam - ``filesystem_usage``, ``listening_sockets`` and
thermal's temperature/battery/fan reads all call ``psutil`` directly. The first
two are canned as models here rather than called. The third is left to answer
for itself: an in-process ``psutil`` read is never a subprocess, a network call
or a mutation, and whichever answer it gives - real sensors on a laptop, an
explicit "unavailable" in a build sandbox - is the availability contract doing
its job and is worth seeing rendered. Adding a seam for any of the three is a
behaviour change and does not belong in an example.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Run from a checkout without installing it. Only the member's `src` is needed:
# this script imports `scufris_host`, never `scufris`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "host" / "src"))

from scufris_host import (  # noqa: E402
    FakeRunner,
    FilesystemReport,
    FilesystemUsage,
    HostInspector,
    ListeningSocket,
    NetworkReport,
    Scope,
    SocketReport,
    StorageReport,
    list_interfaces,
    ok_result,
    profile_contents,
    render,
)

# --- canned command output ---------------------------------------------------

UNITS = json.dumps(
    [
        {
            "unit": "sshd.service",
            "load": "loaded",
            "active": "active",
            "sub": "running",
            "description": "SSH Daemon",
        },
        {
            "unit": "nginx.service",
            "load": "loaded",
            "active": "failed",
            "sub": "failed",
            "description": "Nginx Web Server",
        },
    ]
)

UNIT_SHOW = "\n".join(
    [
        "Id=sshd.service",
        "Description=SSH Daemon",
        "LoadState=loaded",
        "ActiveState=active",
        "SubState=running",
        "UnitFileState=enabled",
        "Result=success",
        "MainPID=1868485",
        "NRestarts=0",
        "MemoryCurrent=7045120",
        "ExecMainStartTimestamp=Wed 2026-07-29 16:31:54 EEST",
    ]
)

# `journalctl -o json` writes one object PER LINE, not an array.
JOURNAL = "\n".join(
    json.dumps(row)
    for row in (
        {
            "MESSAGE": "session opened for user alex",
            "PRIORITY": "6",
            "_PID": "2541431",
            "_SYSTEMD_UNIT": "sshd.service",
            "__REALTIME_TIMESTAMP": "1785331914636340",
        },
        {
            "MESSAGE": "error: could not bind to 0.0.0.0:80",
            "PRIORITY": "3",
            "_PID": "2541999",
            "_SYSTEMD_UNIT": "nginx.service",
            "__REALTIME_TIMESTAMP": "1785331914938999",
        },
    )
)

GENERATIONS = json.dumps(
    [
        {
            "generation": 191,
            "date": "2026-07-29 16:58:12",
            "nixosVersion": "26.11.20260726.624af66",
            "kernelVersion": "6.18.40",
            "configurationRevision": "Unknown",
            "current": True,
        },
        {
            "generation": 190,
            "date": "2026-07-29 16:53:30",
            "nixosVersion": "26.11.20260726.624af66",
            "kernelVersion": "6.18.37",
            "configurationRevision": "Unknown",
            "current": False,
        },
    ]
)

INTERFACES = json.dumps(
    [
        {
            "ifname": "lo",
            "operstate": "UNKNOWN",
            "mtu": 65536,
            "address": "00:00:00:00:00:00",
            "addr_info": [
                {
                    "family": "inet",
                    "local": "127.0.0.1",
                    "prefixlen": 8,
                    "scope": "host",
                }
            ],
        },
        {
            "ifname": "enp5s0",
            "operstate": "UP",
            "mtu": 1500,
            "address": "aa:bb:cc:dd:ee:ff",
            "addr_info": [
                {
                    "family": "inet",
                    "local": "192.168.1.20",
                    "prefixlen": 24,
                    "scope": "global",
                }
            ],
        },
    ]
)

CLOSURE_DIFF = "linux: 6.18.37 -> 6.18.40\nopenssh: 10.4p1 -> 10.5p1\n"

# `du -x -b --max-depth=N -- <root>`: bytes, then a tab, then the path. The row
# for the root itself is the last one and the parser drops it.
DU = "4096000\t{root}/nix\n2048000\t{root}/home\n8192000\t{root}\n"

GC_DRY_RUN = "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-dead-1.0\n"

FIREWALL_SCRIPT = "\n".join(
    [
        "iptables -A nixos-fw -i lo -j nixos-fw-accept",
        "iptables -A nixos-fw -p tcp --dport 22 -j nixos-fw-accept",
        "iptables -A nixos-fw -p tcp --dport 80:443 -j nixos-fw-accept",
        "iptables -A nixos-fw -p udp --dport 51820 -i wg0 -j nixos-fw-accept",
        "iptables -A nixos-fw -j nixos-fw-log-refuse",
    ]
)

FLAKE_LOCK = json.dumps(
    {
        "version": 7,
        "root": "root",
        "nodes": {
            "root": {"inputs": {"nixpkgs": "nixpkgs"}},
            "nixpkgs": {
                "locked": {
                    "type": "github",
                    "owner": "nixos",
                    "repo": "nixpkgs",
                    "rev": "61b7c44c4073f0b827768aff0049561b5110ea5a",
                    "lastModified": 1785000000,
                }
            },
        },
    }
)


def build_runner(du_root: Path) -> FakeRunner:
    """A runner replaying the canned output above, keyed by argv prefix."""
    return FakeRunner(
        results={
            "systemctl --system list-units": ok_result(UNITS),
            "systemctl --system show": ok_result(UNIT_SHOW),
            "journalctl": ok_result(JOURNAL),
            "nixos-rebuild list-generations": ok_result(GENERATIONS),
            "ip -j addr": ok_result(INTERFACES),
            "nix store diff-closures": ok_result(CLOSURE_DIFF),
            "du": ok_result(DU.format(root=du_root)),
            "nix-store --gc": ok_result(GC_DRY_RUN),
        }
    )


def build_system(root: Path) -> Path:
    """A ``/run/current-system`` lookalike whose firewall unit resolves."""
    script = root / "store" / "abc-firewall-start" / "bin" / "firewall-start"
    script.parent.mkdir(parents=True)
    script.write_text(FIREWALL_SCRIPT)
    unit = root / "system" / "etc" / "systemd" / "system" / "firewall.service"
    unit.parent.mkdir(parents=True)
    unit.write_text(f"[Service]\nExecStart={script}\n")
    return root / "system"


def build_cpu_sysfs(root: Path) -> Path:
    """Four CPUs' throttle counters, the shape ``read_throttling`` walks."""
    cpu_sysfs = root / "cpu"
    for cpu in range(4):
        directory = cpu_sysfs / f"cpu{cpu}" / "thermal_throttle"
        directory.mkdir(parents=True)
        (directory / "core_throttle_count").write_text("2\n")
        (directory / "core_throttle_total_time_ms").write_text("5\n")
        (directory / "package_throttle_count").write_text("78\n")
        (directory / "package_throttle_total_time_ms").write_text("153\n")
    return cpu_sysfs


def build_profile(root: Path) -> Path:
    """A system profile with a handful of binaries to list."""
    profile = root / "profile"
    binaries = profile / "bin"
    binaries.mkdir(parents=True)
    for name in ("nix", "systemctl", "journalctl", "ssh"):
        (binaries / name).write_text("#!/bin/sh\n")
    return profile


def build_config_repo(root: Path) -> Path:
    """A NixOS flake directory holding a lock with one pinned input."""
    repo = root / "nix.dotfiles"
    repo.mkdir()
    (repo / "flake.lock").write_text(FLAKE_LOCK)
    return repo


def canned_storage(inspector: HostInspector) -> StorageReport:
    """Generations from the runner, filesystems canned.

    ``storage_report`` would call ``filesystem_usage``, which reads psutil with
    no seam; composing the report here keeps the script off the real machine.
    """
    filesystems = FilesystemReport(
        filesystems=[
            FilesystemUsage(
                mountpoint="/",
                device="/dev/nvme0n1p2",
                fstype="ext4",
                total=500_000_000_000,
                used=310_000_000_000,
                free=190_000_000_000,
                percent=62.0,
            ),
            FilesystemUsage(
                mountpoint="/boot",
                device="/dev/nvme0n1p1",
                fstype="vfat",
                total=1_000_000_000,
                used=910_000_000,
                free=90_000_000,
                percent=91.0,
            ),
        ]
    )
    return StorageReport(
        filesystems=filesystems,
        generations=inspector.generations(),
        nix_store=filesystems.filesystems[0],
    )


def canned_sockets() -> SocketReport:
    """``listening_sockets`` reads psutil with no seam, so this cans its shape."""
    return SocketReport(
        sockets=[
            ListeningSocket(
                protocol="tcp", address="0.0.0.0", port=22, pid=1868485, process="sshd"
            ),
            ListeningSocket(
                protocol="tcp", address="127.0.0.1", port=8080, owner_hidden=True
            ),
            ListeningSocket(
                protocol="udp", address="::", port=51820, pid=421, process="wg"
            ),
        ]
    )


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        system = build_system(root)
        cpu_sysfs = build_cpu_sysfs(root)
        profile = build_profile(root)
        config_repo = build_config_repo(root)
        runner = build_runner(root)
        inspector = HostInspector(
            runner,
            config_repo=config_repo,
            system=system,
            cpu_sysfs=cpu_sysfs,
        )

        section("UNITS")
        print(render.render_units(inspector.list_units(scope=Scope.SYSTEM)))

        section("ONE UNIT")
        print(render.render_unit_status(inspector.unit_status("sshd.service")))

        section("JOURNAL")
        print(render.render_journal(inspector.journal(lines=10)))

        section("STORAGE")
        print(render.render_storage(canned_storage(inspector)))

        section("LARGEST DIRECTORIES")
        print(
            render.render_largest_directories(
                inspector.largest_directories(str(root), depth=1, limit=10)
            )
        )

        section("RECLAIMABLE SPACE")
        print(render.render_reclaimable(inspector.reclaimable_space()))

        section("LISTENING SOCKETS")
        print("\n".join(render.render_sockets(canned_sockets())))

        section("FIREWALL")
        print("\n".join(render.render_firewall(inspector.firewall())))

        section("NETWORK")
        print(
            render.render_network(
                NetworkReport(
                    interfaces=list_interfaces(runner),
                    listening=canned_sockets(),
                    firewall=inspector.firewall(),
                )
            )
        )

        section("THERMAL AND POWER")
        # Throttle counters come from the fixture sysfs above; temperatures,
        # battery and fans have no seam and render their honest "unavailable".
        print(render.render_thermal(inspector.thermal()))

        section("WHAT PROVIDES")
        # Deliberately a name nothing provides: `what_provides` resolves against
        # the real PATH, and the unavailable branch is the one this script can
        # render without depending on what is installed.
        print(render.render_provider(inspector.what_provides("no-such-binary-here")))

        section("SYSTEM PROFILE")
        print(render.render_profile(profile_contents(profile=profile)))

        section("CLOSURE DIFF")
        print(render.render_closure_diff(inspector.closure_diff(190, 191)))

        section("FLAKE INPUTS")
        print(render.render_flake_status(inspector.flake_status()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
