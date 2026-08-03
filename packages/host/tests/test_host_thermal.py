"""Throttling and the thermal report.

Driven against a sysfs tree written by hand, because what is under test is how
logical cpus map onto physical cores: two hyperthread SIBLINGS carry the same
``core_id`` and report the SAME core counter, so a fixture without topology
cannot express the duplication that made the original count double. Covers a
missing sysfs, per-socket separation, an unreadable socket, the package maximum
rather than the sum, a cpu with no topology at all, and the render naming what
it counted.
"""

from __future__ import annotations

from pathlib import Path

from scufris_host import render
from scufris_host.thermal import read_throttling


def test_throttling_says_so_when_sysfs_is_absent(tmp_path: Path) -> None:
    """No counters is "cannot tell", never "it did not throttle"."""
    counters = read_throttling(tmp_path)
    assert not counters.ok
    assert "throttle" in counters.available.reason
    assert counters.core_events == 0


def write_cpu(
    root: Path,
    cpu: int,
    *,
    core_id: int | None,
    core_count: int,
    core_time_ms: int = 10,
    package_count: int = 78,
    package_time_ms: int = 153,
) -> None:
    """Write one logical cpu's thermal_throttle + topology, like real sysfs.

    ``core_id`` is what makes hyperthread SIBLINGS visible: two logical cpus of
    one physical core carry the same id and report the SAME core counter. A
    fixture without it cannot express the duplication, which is exactly why the
    original test agreed with a doubled count.
    """
    directory = root / f"cpu{cpu}" / "thermal_throttle"
    directory.mkdir(parents=True)
    (directory / "core_throttle_count").write_text(f"{core_count}\n")
    (directory / "core_throttle_total_time_ms").write_text(f"{core_time_ms}\n")
    (directory / "package_throttle_count").write_text(f"{package_count}\n")
    (directory / "package_throttle_total_time_ms").write_text(f"{package_time_ms}\n")
    if core_id is not None:
        topology = root / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True, exist_ok=True)
        (topology / "core_id").write_text(f"{core_id}\n")
        (topology / "physical_package_id").write_text("0\n")


def test_throttling_counts_each_physical_core_once(tmp_path: Path) -> None:
    """Hyperthread siblings share a core AND its counter - count it once.

    Reproduces the real shape measured on this host (i9-12900F, SMT on): three
    physical cores had throttled, and their counters appeared on BOTH logical
    cpus of each core, so summing across logical cpus reported exactly 2x.

        cpu8,  cpu9   core_throttle_count=16   <- one core (core_id 16)
        cpu10, cpu11  core_throttle_count=28   <- one core (core_id 20)
        cpu14, cpu15  core_throttle_count=37   <- one core (core_id 28)

    Truth is 16 + 28 + 37 = 81. The bug reported 162.
    """
    idle = [(0, 0), (1, 0), (2, 2), (3, 2), (4, 4), (5, 4)]
    for cpu, core_id in idle:
        write_cpu(tmp_path, cpu, core_id=core_id, core_count=0, core_time_ms=0)
    for cpu, core_id, count, ms in (
        (8, 16, 16, 37),
        (9, 16, 16, 37),
        (10, 20, 28, 67),
        (11, 20, 28, 67),
        (14, 28, 37, 51),
        (15, 28, 37, 51),
    ):
        write_cpu(tmp_path, cpu, core_id=core_id, core_count=count, core_time_ms=ms)

    counters = read_throttling(tmp_path)
    assert counters.ok
    # 16 + 28 + 37, each counted ONCE - not 162.
    assert counters.core_events == 81
    # The time counter is duplicated identically and must not double either.
    assert counters.core_time_ms == 37 + 67 + 51
    # 6 physical cores (3 idle pairs + 3 throttled pairs) behind 12 logical cpus.
    assert counters.cores_read == 6
    assert counters.cpus_read == 12
    assert counters.throttled


def test_throttling_takes_the_higher_of_two_disagreeing_siblings(
    tmp_path: Path,
) -> None:
    """Siblings can be momentarily out of step, so reduce with MAX.

    Each cpu's own thermal-interrupt handler writes these counters, and the
    package counters demonstrably skew on the real host (78 on most cpus, 80 on
    two, 82 on two). Last-write-wins would silently keep the stale sibling, and
    WHICH one wins depends on glob order - where "cpu10" sorts before "cpu2".
    """
    write_cpu(tmp_path, 0, core_id=4, core_count=37, core_time_ms=51)
    write_cpu(tmp_path, 1, core_id=4, core_count=36, core_time_ms=50)
    counters = read_throttling(tmp_path)
    assert counters.ok
    assert counters.cores_read == 1
    assert counters.core_events == 37, "the stale sibling won"
    assert counters.core_time_ms == 51


def test_throttling_keeps_same_numbered_cores_of_different_sockets_apart(
    tmp_path: Path,
) -> None:
    """`core_id` is unique only WITHIN a package, so the socket must be in the key.

    Two sockets each have a core 0. Keying on core_id alone would merge them and
    report one core - an undercount.
    """
    for cpu, package in ((0, 0), (1, 1)):
        directory = tmp_path / f"cpu{cpu}" / "thermal_throttle"
        directory.mkdir(parents=True)
        (directory / "core_throttle_count").write_text("5\n")
        (directory / "core_throttle_total_time_ms").write_text("10\n")
        (directory / "package_throttle_count").write_text("78\n")
        (directory / "package_throttle_total_time_ms").write_text("153\n")
        topology = tmp_path / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "core_id").write_text("0\n")
        (topology / "physical_package_id").write_text(f"{package}\n")
    counters = read_throttling(tmp_path)
    assert counters.cores_read == 2, "two sockets' core 0 were merged"
    assert counters.core_events == 10


def test_throttling_does_not_merge_cores_when_the_socket_is_unreadable(
    tmp_path: Path,
) -> None:
    """A readable core_id with an unreadable package_id must not merge sockets.

    The fallback direction matters: over-counting cores is recoverable, an
    undercount silently hides throttling.
    """
    for cpu in (0, 1):
        directory = tmp_path / f"cpu{cpu}" / "thermal_throttle"
        directory.mkdir(parents=True)
        (directory / "core_throttle_count").write_text("5\n")
        (directory / "core_throttle_total_time_ms").write_text("10\n")
        (directory / "package_throttle_count").write_text("78\n")
        (directory / "package_throttle_total_time_ms").write_text("153\n")
        topology = tmp_path / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "core_id").write_text("0\n")  # no physical_package_id
    counters = read_throttling(tmp_path)
    assert counters.cores_read == 2
    assert counters.core_events == 10


def test_throttling_reports_how_many_cores_actually_throttled(
    tmp_path: Path,
) -> None:
    """Concentration is the interesting part: 81 events on 3 of 16 cores."""
    for cpu in range(4):
        write_cpu(tmp_path, cpu, core_id=cpu, core_count=0, core_time_ms=0)
    write_cpu(tmp_path, 4, core_id=4, core_count=37)
    counters = read_throttling(tmp_path)
    assert counters.cores_read == 5
    assert counters.cores_throttled == 1


def test_thermal_render_names_what_it_counted(tmp_path: Path) -> None:
    """The rendered sentence must say WHICH unit each figure is counted in.

    The Definition-of-Done proof for the wording. "162 core events across 24
    cpus" was the original defect's second half: even with the arithmetic fixed,
    quoting a per-core figure against a logical-cpu count invites dividing by the
    wrong denominator.
    """
    from scufris_host.thermal import ThermalReport, ThrottleCounters

    throttled = ThermalReport(
        throttling=ThrottleCounters(
            core_events=81,
            core_time_ms=155,
            package_events=82,
            package_time_ms=153,
            cpus_read=24,
            cores_read=16,
            cores_throttled=3,
        )
    )
    text = render.render_thermal(throttled)
    assert "per-core events" in text
    assert "whole-package events" in text
    assert "3 of 16 physical cores" in text
    # The bare, ambiguous phrasing must be gone.
    assert "core events across" not in text
    assert "across 24 cpus" not in text

    quiet = ThermalReport(
        throttling=ThrottleCounters(cpus_read=24, cores_read=16, cores_throttled=0)
    )
    quiet_text = render.render_thermal(quiet)
    assert "no thermal throttling recorded" in quiet_text
    assert "physical cores" in quiet_text
    assert "logical cpus" in quiet_text


def test_throttling_takes_the_package_maximum_not_the_sum(tmp_path: Path) -> None:
    """The package half, which was already right and must stay right.

    Package counters are per-package, so summing multiplies by cpu count. On the
    real host they are also not perfectly identical (78 on most cpus, 82 on two)
    because each cpu updates its own view when it handles the thermal interrupt -
    so max is the right reduction for FRESHNESS too, not only for deduplication.
    """
    for cpu, package_count in ((0, 78), (1, 78), (2, 82), (3, 80)):
        write_cpu(tmp_path, cpu, core_id=cpu, core_count=5, package_count=package_count)
    counters = read_throttling(tmp_path)
    assert counters.ok
    assert counters.package_events == 82  # NOT 318, and not 78
    assert counters.package_time_ms == 153
    assert counters.core_events == 20  # four distinct cores here


def test_throttling_counts_a_cpu_with_no_topology(tmp_path: Path) -> None:
    """No topology/core_id must not DROP a cpu - under-counting is still wrong.

    A container, a non-x86 host or an older kernel may expose the throttle
    counters without the topology beside them. Falling back to the cpu's own
    identity keeps it counted; silently skipping it would turn a missing file
    into a smaller number, which is the failure this whole package exists to
    avoid.
    """
    write_cpu(tmp_path, 0, core_id=None, core_count=7)
    write_cpu(tmp_path, 1, core_id=None, core_count=9)
    counters = read_throttling(tmp_path)
    assert counters.ok
    assert counters.core_events == 16, "a cpu without topology was dropped"
    assert counters.cores_read == 2
    assert counters.cpus_read == 2
