"""Host metrics collection.

A small, read-only snapshot of the host the app runs on. Collection sits behind
the ``Collector`` protocol so the source is swappable and tests can fake the
seam rather than patching psutil internals. ``PsutilCollector`` is the real
implementation; ``sample()`` never blocks (CPU percent is read as a non-blocking
delta primed at construction).

GPU stats come from the ``nvidia-smi`` CLI (behind an injectable runner), not an
NVML Python binding - the CLI is robust on NixOS and needs no driver linkage
(see tasks/20260719-180507/SPIKE.md). Network and disk figures are reported as
per-second RATES computed from counters persisted between samples.
"""

from __future__ import annotations

import platform
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable

import psutil
from pydantic import BaseModel, Field


class MemStats(BaseModel):
    total: int
    used: int
    available: int
    percent: float


class SwapStats(BaseModel):
    total: int
    used: int
    percent: float


class DiskUsage(BaseModel):
    mountpoint: str
    total: int
    used: int
    percent: float


class NetIO(BaseModel):
    bytes_sent: int
    bytes_recv: int


class GpuStats(BaseModel):
    name: str
    util_percent: float
    mem_used_mb: int
    mem_total_mb: int
    mem_percent: float
    temp_c: float
    power_w: float
    power_limit_w: float
    clock_sm_mhz: int
    clock_mem_mhz: int


class SensorReading(BaseModel):
    label: str
    current: float
    high: float | None = None
    critical: float | None = None


class SensorGroup(BaseModel):
    chip: str
    readings: list[SensorReading]


class FanReading(BaseModel):
    chip: str
    label: str
    rpm: int


class NetIfRate(BaseModel):
    name: str
    sent_per_sec: float
    recv_per_sec: float


class DiskIoRate(BaseModel):
    name: str
    read_per_sec: float
    write_per_sec: float


class CpuActivity(BaseModel):
    ctx_switches_per_sec: float = 0.0
    interrupts_per_sec: float = 0.0


class HostStats(BaseModel):
    """A single read-only snapshot of host metrics."""

    hostname: str
    os_name: str
    kernel: str
    cpu_percent: float
    per_cpu_percent: list[float]
    mem: MemStats
    swap: SwapStats
    disks: list[DiskUsage]
    load_avg: tuple[float, float, float]
    uptime_seconds: float
    net: NetIO
    sampled_at: datetime
    # Richer detail (empty when unavailable, so older constructions still work).
    gpus: list[GpuStats] = Field(default_factory=list)
    temps: list[SensorGroup] = Field(default_factory=list)
    fans: list[FanReading] = Field(default_factory=list)
    per_cpu_freq_mhz: list[float] = Field(default_factory=list)
    net_interfaces: list[NetIfRate] = Field(default_factory=list)
    disk_io: list[DiskIoRate] = Field(default_factory=list)
    process_count: int = 0
    cpu_activity: CpuActivity = Field(default_factory=CpuActivity)


@runtime_checkable
class Collector(Protocol):
    """The seam the backend depends on. Tests provide a fake implementation."""

    def sample(self) -> HostStats:
        """Return a fresh snapshot of the host's metrics."""
        ...


# GPU query columns, in order, matching GpuStats after the name.
_GPU_QUERY = (
    "name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
    "power.draw,power.limit,clocks.sm,clocks.mem"
)

# A GPU runner returns raw nvidia-smi CSV text (or None when unavailable). It is
# injectable so tests can feed a captured sample without a GPU.
GpuRunner = Callable[[], "str | None"]


def _run_nvidia_smi() -> str | None:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        proc = subprocess.run(
            [smi, f"--query-gpu={_GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        # nvidia-smi reports "[N/A]" for unsupported fields.
        return 0.0


def parse_gpus(csv_text: str | None) -> list[GpuStats]:
    """Parse ``nvidia-smi --format=csv,noheader,nounits`` output into GpuStats."""
    gpus: list[GpuStats] = []
    if not csv_text:
        return gpus
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        name = parts[0]
        util, used, total, temp, power, limit, sm, mem = (
            _to_float(p) for p in parts[1:9]
        )
        gpus.append(
            GpuStats(
                name=name,
                util_percent=util,
                mem_used_mb=int(used),
                mem_total_mb=int(total),
                mem_percent=(used / total * 100.0) if total > 0 else 0.0,
                temp_c=temp,
                power_w=power,
                power_limit_w=limit,
                clock_sm_mhz=int(sm),
                clock_mem_mhz=int(mem),
            )
        )
    return gpus


class PsutilCollector:
    """Collect host metrics via psutil (+ nvidia-smi for GPU).

    CPU percent and net/disk rates are deltas since the previous read, so the
    collector is stateful and a single instance must be reused across samples.
    The first read after construction primes CPU percent; the first ``sample()``
    reports zero net/disk rates (no prior counters yet).
    """

    def __init__(self, gpu_runner: GpuRunner = _run_nvidia_smi) -> None:
        self._hostname = socket.gethostname()
        self._os_name = platform.system()
        self._kernel = platform.release()
        self._gpu_runner = gpu_runner
        # Prime the non-blocking CPU counters so the first sample() is real.
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        # Previous counters + monotonic timestamp for rate computation.
        self._prev_net: dict[str, Any] | None = None
        self._prev_disk: dict[str, Any] | None = None
        self._prev_mono: float | None = None
        self._prev_cpu_stats: Any | None = None
        self._prev_cpu_time: float | None = None

    def sample(self) -> HostStats:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        net = psutil.net_io_counters()
        net_rates, disk_rates = self._io_rates()
        return HostStats(
            hostname=self._hostname,
            os_name=self._os_name,
            kernel=self._kernel,
            cpu_percent=psutil.cpu_percent(interval=None),
            per_cpu_percent=psutil.cpu_percent(interval=None, percpu=True),
            mem=MemStats(
                total=vm.total,
                used=vm.used,
                available=vm.available,
                percent=vm.percent,
            ),
            swap=SwapStats(total=sm.total, used=sm.used, percent=sm.percent),
            disks=self._disks(),
            load_avg=self._load_avg(),
            uptime_seconds=max(0.0, time.time() - psutil.boot_time()),
            net=NetIO(bytes_sent=net.bytes_sent, bytes_recv=net.bytes_recv),
            sampled_at=datetime.now(timezone.utc),
            gpus=parse_gpus(self._gpu_runner()),
            temps=self._temps(),
            fans=self._fans(),
            per_cpu_freq_mhz=self._per_cpu_freq(),
            net_interfaces=net_rates,
            disk_io=disk_rates,
            process_count=len(psutil.pids()),
            cpu_activity=self._cpu_activity(),
        )

    def _cpu_activity(self) -> CpuActivity:
        now = time.monotonic()
        stats = psutil.cpu_stats()
        ctx = 0.0
        interrupts = 0.0
        prev, prev_time = self._prev_cpu_stats, self._prev_cpu_time
        if prev is not None and prev_time is not None and now > prev_time:
            dt = now - prev_time
            ctx = max(0.0, stats.ctx_switches - prev.ctx_switches) / dt
            interrupts = max(0.0, stats.interrupts - prev.interrupts) / dt
        self._prev_cpu_stats = stats
        self._prev_cpu_time = now
        return CpuActivity(ctx_switches_per_sec=ctx, interrupts_per_sec=interrupts)

    def _io_rates(self) -> tuple[list[NetIfRate], list[DiskIoRate]]:
        now = time.monotonic()
        net = psutil.net_io_counters(pernic=True)
        disk = psutil.disk_io_counters(perdisk=True) or {}
        net_rates: list[NetIfRate] = []
        disk_rates: list[DiskIoRate] = []
        prev_net, prev_disk, prev_mono = (
            self._prev_net,
            self._prev_disk,
            self._prev_mono,
        )
        if prev_net is not None and prev_mono is not None and now > prev_mono:
            dt = now - prev_mono
            for name, cur in net.items():
                prev = prev_net.get(name)
                if prev is None:
                    continue
                net_rates.append(
                    NetIfRate(
                        name=name,
                        sent_per_sec=max(0.0, cur.bytes_sent - prev.bytes_sent) / dt,
                        recv_per_sec=max(0.0, cur.bytes_recv - prev.bytes_recv) / dt,
                    )
                )
            if prev_disk is not None:
                for dname, dcur in disk.items():
                    dprev = prev_disk.get(dname)
                    if dprev is None:
                        continue
                    disk_rates.append(
                        DiskIoRate(
                            name=dname,
                            read_per_sec=max(0.0, dcur.read_bytes - dprev.read_bytes)
                            / dt,
                            write_per_sec=max(0.0, dcur.write_bytes - dprev.write_bytes)
                            / dt,
                        )
                    )
        self._prev_net = net
        self._prev_disk = disk
        self._prev_mono = now
        net_rates.sort(key=lambda r: r.sent_per_sec + r.recv_per_sec, reverse=True)
        disk_rates.sort(key=lambda r: r.read_per_sec + r.write_per_sec, reverse=True)
        return net_rates, disk_rates

    @staticmethod
    def _temps() -> list[SensorGroup]:
        try:
            raw = psutil.sensors_temperatures()
        except (AttributeError, OSError):
            return []
        groups: list[SensorGroup] = []
        for chip, entries in raw.items():
            readings = [
                SensorReading(
                    label=e.label or chip,
                    current=e.current,
                    high=e.high,
                    critical=e.critical,
                )
                for e in entries
            ]
            groups.append(SensorGroup(chip=chip, readings=readings))
        return groups

    @staticmethod
    def _fans() -> list[FanReading]:
        try:
            raw = psutil.sensors_fans()
        except (AttributeError, OSError):
            return []
        fans: list[FanReading] = []
        for chip, entries in raw.items():
            for e in entries:
                fans.append(FanReading(chip=chip, label=e.label or chip, rpm=e.current))
        return fans

    @staticmethod
    def _per_cpu_freq() -> list[float]:
        try:
            freqs = psutil.cpu_freq(percpu=True)
        except (NotImplementedError, AttributeError, OSError):
            return []
        return [f.current for f in (freqs or [])]

    @staticmethod
    def _disks() -> list[DiskUsage]:
        disks: list[DiskUsage] = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except (PermissionError, OSError):
                # A mount we cannot stat (removable, restricted) is skipped
                # rather than failing the whole sample.
                continue
            disks.append(
                DiskUsage(
                    mountpoint=part.mountpoint,
                    total=usage.total,
                    used=usage.used,
                    percent=usage.percent,
                )
            )
        return disks

    @staticmethod
    def _load_avg() -> tuple[float, float, float]:
        try:
            one, five, fifteen = psutil.getloadavg()
        except (AttributeError, OSError):
            # getloadavg is unavailable on some platforms; report zeros.
            return (0.0, 0.0, 0.0)
        return (one, five, fifteen)
