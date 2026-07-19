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
    parse_gpus,
)

# A real captured line from `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`.
_GPU_CSV = "NVIDIA GeForce RTX 3060 Ti, 4, 362, 8192, 38, 10.28, 225.00, 210, 405"


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
    # Richer fields are always lists (populated on hosts that expose them).
    assert isinstance(stats.per_cpu_freq_mhz, list)
    assert isinstance(stats.temps, list)


def test_parse_gpus_from_sample() -> None:
    gpus = parse_gpus(_GPU_CSV)
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.name == "NVIDIA GeForce RTX 3060 Ti"
    assert gpu.util_percent == 4.0
    assert gpu.mem_used_mb == 362
    assert gpu.mem_total_mb == 8192
    assert 4.0 < gpu.mem_percent < 5.0
    assert gpu.temp_c == 38.0
    assert gpu.power_w == 10.28
    assert gpu.clock_sm_mhz == 210


def test_parse_gpus_handles_missing_or_malformed() -> None:
    assert parse_gpus(None) == []
    assert parse_gpus("") == []
    assert parse_gpus("too, few, cols") == []
    # "[N/A]" fields degrade to 0.0 rather than raising.
    na = parse_gpus("GpuX, [N/A], 0, 0, [N/A], [N/A], [N/A], [N/A], [N/A]")
    assert na[0].util_percent == 0.0
    assert na[0].mem_percent == 0.0


def test_collector_uses_injected_gpu_runner() -> None:
    collector = PsutilCollector(gpu_runner=lambda: _GPU_CSV)
    stats = collector.sample()
    assert len(stats.gpus) == 1
    assert stats.gpus[0].name == "NVIDIA GeForce RTX 3060 Ti"


def test_net_disk_rates_appear_on_second_sample() -> None:
    collector = PsutilCollector(gpu_runner=lambda: None)
    first = collector.sample()
    # No previous counters yet, so the first sample carries no rates.
    assert first.net_interfaces == []
    second = collector.sample()
    # Loopback always exists, so the second sample has at least one interface.
    assert len(second.net_interfaces) >= 1
    assert all(r.sent_per_sec >= 0.0 for r in second.net_interfaces)
