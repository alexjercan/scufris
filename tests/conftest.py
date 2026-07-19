"""Shared test fixtures.

`FakeCollector` returns a deterministic `HostStats`, so backend/API tests never
touch real host state or psutil.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scufris.metrics import (
    DiskUsage,
    HostStats,
    MemStats,
    NetIO,
    SwapStats,
)


class FakeCollector:
    def __init__(self, stats: HostStats) -> None:
        self._stats = stats

    def sample(self) -> HostStats:
        return self._stats


def make_fixture_stats() -> HostStats:
    return HostStats(
        hostname="testbox",
        os_name="Linux",
        kernel="6.18.0",
        cpu_percent=12.5,
        per_cpu_percent=[10.0, 15.0],
        mem=MemStats(total=1000, used=400, available=600, percent=40.0),
        swap=SwapStats(total=200, used=50, percent=25.0),
        disks=[DiskUsage(mountpoint="/", total=500, used=100, percent=20.0)],
        load_avg=(0.1, 0.2, 0.3),
        uptime_seconds=1234.0,
        net=NetIO(bytes_sent=10, bytes_recv=20),
        sampled_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def fake_stats() -> HostStats:
    return make_fixture_stats()


@pytest.fixture
def fake_collector(fake_stats: HostStats) -> FakeCollector:
    return FakeCollector(fake_stats)
