"""Host metrics collection.

A small, read-only snapshot of the host the app runs on. Collection sits behind
the ``Collector`` protocol so the source is swappable and tests can fake the
seam rather than patching psutil internals. ``PsutilCollector`` is the real
implementation; ``sample()`` never blocks (CPU percent is read as a non-blocking
delta primed at construction).
"""

from __future__ import annotations

import platform
import socket
import time
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import psutil
from pydantic import BaseModel


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


@runtime_checkable
class Collector(Protocol):
    """The seam the backend depends on. Tests provide a fake implementation."""

    def sample(self) -> HostStats:
        """Return a fresh snapshot of the host's metrics."""
        ...


class PsutilCollector:
    """Collect host metrics via psutil.

    CPU percent is measured as the delta since the previous read. The first
    read after a process starts always reports 0.0, so we prime it in the
    constructor; ``sample()`` then returns a meaningful non-blocking value.
    """

    def __init__(self) -> None:
        self._hostname = socket.gethostname()
        self._os_name = platform.system()
        self._kernel = platform.release()
        # Prime the non-blocking CPU counters so the first sample() is real.
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)

    def sample(self) -> HostStats:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        net = psutil.net_io_counters()
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
        )

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
