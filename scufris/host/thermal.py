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
    cpus_read: int = 0

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


def read_throttling(cpu_sysfs: Path = CPU_SYSFS) -> ThrottleCounters:
    """Sum the per-CPU thermal throttle counters.

    Per-core counts are summed across CPUs; the PACKAGE counters are per-package
    and identical on every CPU of that package, so summing them would multiply
    by core count - the maximum is taken instead.
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
    for directory in directories:
        core_count = _read_int(directory / "core_throttle_count")
        if core_count is None:
            continue
        counters.cpus_read += 1
        counters.core_events += core_count
        counters.core_time_ms += (
            _read_int(directory / "core_throttle_total_time_ms") or 0
        )
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
