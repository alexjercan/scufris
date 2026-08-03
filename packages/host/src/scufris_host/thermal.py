"""Thermal and power inspection: the data that answers "why is this box hot".

A temperature gauge alone cannot answer that question, because the interesting
case is the one where the CPU is being held back and the temperature therefore
looks fine. The signal that actually settles it is the kernel's throttle
counters under ``/sys/devices/system/cpu/cpu*/thermal_throttle/`` - this host has
recorded 78 package throttle events while sitting at 31C. Those counters are the
reason this module exists rather than the existing ``metrics.HostStats.temps``.

Battery and fans are IMPLEMENTED, not omitted, even though this host has
neither: ``/sys/class/power_supply/`` is empty, ``psutil.sensors_battery()``
returns None and ``sensors_fans()`` returns ``{}``. They report "not present on
this host" explicitly, which is a real answer and keeps the module honest if it
ever runs on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import psutil
from pydantic import BaseModel, Field

from .models import Availability, Report

CPU_SYSFS = Path("/sys/devices/system/cpu")

# Rising within this margin of the trip point is the interesting "about to
# throttle" state that a raw number does not convey.
_NEAR_LIMIT_MARGIN_C = 10.0


class TemperatureReading(BaseModel):
    chip: str
    label: str
    celsius: float
    high: float | None = None
    critical: float | None = None

    @property
    def limit(self) -> float | None:
        """The trip point that matters: critical if set, else the high mark.

        `is not None`, not truthiness: a sensor legitimately reporting 0.0 would
        otherwise silently fall through to `high`.
        """
        return self.critical if self.critical is not None else self.high

    @property
    def near_limit(self) -> bool:
        limit = self.limit
        return limit is not None and self.celsius >= limit - _NEAR_LIMIT_MARGIN_C

    @property
    def over_limit(self) -> bool:
        limit = self.limit
        return limit is not None and self.celsius >= limit


class ThrottleCounters(Report):
    """Kernel-recorded thermal throttling, cumulative since boot.

    A non-zero count is proof the CPU was actually held back, which no
    temperature reading can show after the fact.
    """

    core_events: int = 0
    package_events: int = 0
    core_time_ms: int = 0
    package_time_ms: int = 0
    # Logical cpus whose counters were readable, and the PHYSICAL cores behind
    # them. Both are reported because the difference is the whole subtlety here:
    # `core_events` is per physical core, so quoting it "across N cpus" invites
    # the reader to divide by the wrong number.
    cpus_read: int = 0
    cores_read: int = 0
    # How many of `cores_read` actually threw an event. "81 across 16 cores"
    # reads as a distribution over 16; on this host only 3 cores ever throttled,
    # and that concentration is the interesting part.
    cores_throttled: int = 0

    @property
    def throttled(self) -> bool:
        return self.core_events > 0 or self.package_events > 0


class BatteryState(Report):
    present: bool = False
    percent: float | None = None
    plugged_in: bool | None = None
    seconds_left: int | None = None


class FanReport(Report):
    fans: list[TemperatureReading] = Field(default_factory=list)
    present: bool = False


class ThermalReport(Report):
    temperatures: list[TemperatureReading] = Field(default_factory=list)
    throttling: ThrottleCounters = Field(default_factory=ThrottleCounters)
    battery: BatteryState = Field(default_factory=BatteryState)
    fans: FanReport = Field(default_factory=FanReport)

    @property
    def hottest(self) -> TemperatureReading | None:
        return max(self.temperatures, key=lambda t: t.celsius, default=None)

    @property
    def concerning(self) -> list[TemperatureReading]:
        return [t for t in self.temperatures if t.near_limit]


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def read_temperatures() -> tuple[list[TemperatureReading], Availability]:
    """Every temperature sensor psutil can see, with its trip points."""
    try:
        raw = psutil.sensors_temperatures()
    except (AttributeError, OSError) as exc:
        return [], Availability.unavailable(
            f"temperature sensors could not be read: {exc}"
        )
    readings = [
        TemperatureReading(
            chip=chip,
            label=entry.label or chip,
            celsius=entry.current,
            high=entry.high,
            critical=entry.critical,
        )
        for chip, entries in raw.items()
        for entry in entries
    ]
    if not readings:
        return [], Availability.unavailable(
            "this host exposes no temperature sensors (no hwmon/coretemp data)"
        )
    readings.sort(key=lambda t: t.celsius, reverse=True)
    return readings, Availability()


def _core_identity(cpu_directory: Path) -> str:
    """A key that is the same for two hyperthread siblings, different otherwise.

    Read from ``topology/`` next to the counters. When the topology is absent -
    a container, a non-x86 host, an older kernel - the cpu's own directory name
    is the fallback, so it counts as its own core. That direction is the safe
    one: an unknown topology must never make a cpu vanish from the total.
    """
    topology = cpu_directory / "topology"
    core_id = _read_int(topology / "core_id")
    package_id = _read_int(topology / "physical_package_id")
    if core_id is None or package_id is None:
        # `core_id` is only unique WITHIN a package, so a readable core_id with
        # an unreadable package_id would merge core 0 of socket 0 with core 0 of
        # socket 1 - an UNDERCOUNT, the direction this fallback exists to
        # prevent. Falling back to the cpu's own name over-counts at worst,
        # which is the safe way to be wrong here.
        return f"cpu:{cpu_directory.name}"
    return f"pkg{package_id}/core{core_id}"


def read_throttling(cpu_sysfs: Path = CPU_SYSFS) -> ThrottleCounters:
    """Read the thermal throttle counters, deduplicated to physical hardware.

    Both counter families are duplicated in sysfs, at different levels, and each
    needs its own reduction:

    - CORE counters appear once per LOGICAL cpu, and hyperthread siblings of one
      physical core report the SAME value. They are summed over distinct cores,
      keyed by ``topology/core_id``. Summing over logical cpus instead reports
      exactly 2x on an SMT machine - measured on this host, 162 where the truth
      was 81.
    - PACKAGE counters appear on every cpu of a package, so the MAXIMUM is taken.
      Note they are not always perfectly identical (78 on most cpus, 82 on two,
      measured here) because each cpu updates its own view when it handles the
      thermal interrupt - so max is also the freshest reading, not merely the
      deduplicated one. Do not "simplify" this to a sum.
    """
    counters = ThrottleCounters()
    if not cpu_sysfs.is_dir():
        counters.available = Availability.unavailable(
            f"{cpu_sysfs} does not exist, so throttling cannot be determined "
            "(this reads Linux CPU sysfs)"
        )
        return counters
    directories = sorted(cpu_sysfs.glob("cpu[0-9]*/thermal_throttle"))
    if not directories:
        counters.available = Availability.unavailable(
            "this CPU does not expose thermal_throttle counters, so whether it "
            "throttled cannot be determined from sysfs"
        )
        return counters
    package_events = 0
    package_time = 0
    # One entry per PHYSICAL core. Siblings are reduced with MAX rather than
    # summed (which double-counts) or last-write-wins (which is arbitrary):
    # these counters are written by each cpu's own thermal-interrupt handler, so
    # two siblings can be momentarily out of step. The package counters prove
    # that skew is real on this host - they read 78 on most cpus, 80 on two and
    # 82 on two - so the same reduction applies one level down. Last-write-wins
    # would additionally depend on glob order, where "cpu10" sorts before
    # "cpu2".
    by_core: dict[str, tuple[int, int]] = {}
    for directory in directories:
        core_count = _read_int(directory / "core_throttle_count")
        if core_count is None:
            continue
        counters.cpus_read += 1
        core_time = _read_int(directory / "core_throttle_total_time_ms") or 0
        key = _core_identity(directory.parent)
        seen_count, seen_time = by_core.get(key, (0, 0))
        by_core[key] = (max(seen_count, core_count), max(seen_time, core_time))
        package_events = max(
            package_events, _read_int(directory / "package_throttle_count") or 0
        )
        package_time = max(
            package_time, _read_int(directory / "package_throttle_total_time_ms") or 0
        )
    if counters.cpus_read == 0:
        counters.available = Availability.unavailable(
            "the thermal_throttle counters exist but none could be read"
        )
        return counters
    counters.cores_read = len(by_core)
    counters.cores_throttled = sum(1 for count, _ in by_core.values() if count)
    counters.core_events = sum(count for count, _ in by_core.values())
    counters.core_time_ms = sum(millis for _, millis in by_core.values())
    counters.package_events = package_events
    counters.package_time_ms = package_time
    return counters


def read_battery() -> BatteryState:
    """Battery state, or an explicit "no battery" for a desktop.

    Measured on this host: no battery, which is a real answer and not a failure.
    """
    try:
        battery = psutil.sensors_battery()
    except (AttributeError, OSError) as exc:
        return BatteryState(
            available=Availability.unavailable(f"battery state is unreadable: {exc}")
        )
    if battery is None:
        return BatteryState(
            present=False,
            available=Availability(
                ok=True, caveat="this host has no battery (it is not a laptop)"
            ),
        )
    seconds = battery.secsleft
    return BatteryState(
        present=True,
        percent=battery.percent,
        plugged_in=battery.power_plugged,
        seconds_left=(
            int(seconds)
            if seconds not in (psutil.POWER_TIME_UNLIMITED, psutil.POWER_TIME_UNKNOWN)
            else None
        ),
    )


def read_fans() -> FanReport:
    """Fan speeds, or an explicit "no fan sensors" for a host that exposes none."""
    try:
        raw = psutil.sensors_fans()
    except (AttributeError, OSError) as exc:
        return FanReport(
            available=Availability.unavailable(f"fan sensors are unreadable: {exc}")
        )
    fans = [
        TemperatureReading(chip=chip, label=entry.label or chip, celsius=entry.current)
        for chip, entries in raw.items()
        for entry in entries
    ]
    if not fans:
        return FanReport(
            present=False,
            available=Availability(
                ok=True,
                caveat="this host exposes no fan sensors, so fan speed is unknown",
            ),
        )
    return FanReport(present=True, fans=fans)


def thermal_report(cpu_sysfs: Path = CPU_SYSFS) -> ThermalReport:
    """Temperatures, throttling, battery and fans in one answer."""
    temperatures, availability = read_temperatures()
    report = ThermalReport(
        temperatures=temperatures,
        throttling=read_throttling(cpu_sysfs),
        battery=read_battery(),
        fans=read_fans(),
    )
    if not availability.ok and not report.throttling.ok:
        # Neither half of the thermal answer is available; say so rather than
        # returning an empty report that reads as "nothing is hot".
        report.available = Availability.unavailable(
            f"{availability.reason}; {report.throttling.available.reason}"
        )
    elif not availability.ok:
        report.available = Availability(ok=True, caveat=availability.reason)
    return report
