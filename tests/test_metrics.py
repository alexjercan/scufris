"""Tests for the host metrics collector.

The fake-collector test pins the model shape and the ``Collector`` seam without
touching psutil; the smoke test proves the real ``PsutilCollector`` populates a
snapshot on this host.
"""

from __future__ import annotations

from datetime import datetime, timezone

from scufris.metrics import (
    Collector,
    DiskUsage,
    HostStats,
    MemStats,
    NetIO,
    PsutilCollector,
    SwapStats,
)


def _fixture_stats() -> HostStats:
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


class FakeCollector:
    """A deterministic collector used by the backend tests too."""

    def __init__(self, stats: HostStats) -> None:
        self._stats = stats

    def sample(self) -> HostStats:
        return self._stats


def test_fake_collector_satisfies_protocol_and_serializes() -> None:
    stats = _fixture_stats()
    collector: Collector = FakeCollector(stats)

    assert isinstance(collector, Collector)

    result = collector.sample()
    assert result is stats

    # The snapshot must round-trip through JSON so the API can serve it.
    payload = result.model_dump(mode="json")
    assert payload["hostname"] == "testbox"
    assert payload["mem"]["percent"] == 40.0
    assert payload["disks"][0]["mountpoint"] == "/"
    assert payload["load_avg"] == [0.1, 0.2, 0.3]


def test_psutil_collector_populates_a_snapshot() -> None:
    stats = PsutilCollector().sample()

    assert isinstance(stats, HostStats)
    assert stats.hostname
    assert stats.os_name
    assert 0.0 <= stats.cpu_percent <= 100.0
    assert len(stats.per_cpu_percent) >= 1
    assert stats.mem.total > 0
    assert stats.mem.used >= 0
    assert stats.uptime_seconds >= 0.0
    assert stats.net.bytes_recv >= 0
    assert stats.sampled_at.tzinfo is not None
