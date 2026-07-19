"""Tests for process aggregation and the psutil process collector."""

from __future__ import annotations

from scufris.processes import (
    ProcessInstance,
    ProcessList,
    PsutilProcessCollector,
    aggregate_processes,
)


def _inst(cpu: float, mem: int, pid: int = 1) -> ProcessInstance:
    return ProcessInstance(
        pid=pid,
        username="alex",
        cpu_percent=cpu,
        mem_rss=mem,
        num_threads=1,
        status="running",
    )


def test_aggregate_groups_by_name_and_sums() -> None:
    rows = [
        ("firefox", _inst(10, 100, 1)),
        ("firefox", _inst(20, 200, 2)),
        ("rustc", _inst(50, 50, 3)),
    ]
    result = aggregate_processes(rows)
    assert isinstance(result, ProcessList)
    assert result.total == 3
    # Sorted by cpu desc: rustc (50) before firefox (30).
    assert [g.name for g in result.groups] == ["rustc", "firefox"]
    firefox = result.groups[1]
    assert firefox.count == 2
    assert firefox.cpu_percent == 30.0
    assert firefox.mem_rss == 300
    # Instances within a group are sorted by cpu desc.
    assert [i.cpu_percent for i in firefox.instances] == [20.0, 10.0]


def test_aggregate_caps_groups_and_instances() -> None:
    rows = [
        ("a", _inst(90, 1, 1)),
        ("b", _inst(80, 1, 2)),
        ("c", _inst(70, 1, 3)),
    ]
    top2 = aggregate_processes(rows, top_groups=2)
    assert [g.name for g in top2.groups] == ["a", "b"]
    assert top2.total == 3  # total counts everything, not just the survivors

    many = [("x", _inst(float(i), 1, i)) for i in range(10)]
    capped = aggregate_processes(many, top_instances=3)
    group = capped.groups[0]
    assert group.count == 10  # count covers all
    assert len(group.instances) == 3  # instances are capped to the top 3
    assert [i.cpu_percent for i in group.instances] == [9.0, 8.0, 7.0]


def test_psutil_process_collector_populates() -> None:
    result = PsutilProcessCollector().sample()
    assert isinstance(result, ProcessList)
    assert result.total > 0
    assert len(result.groups) >= 1
    assert all(g.count >= 1 for g in result.groups)
