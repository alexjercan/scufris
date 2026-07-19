"""Process listing, aggregated by application.

The heavy per-process feed is separate from the light ``/api/stats`` snapshot.
Processes are grouped by name (application) with summed cpu/mem, a count, and the
top instances - the "btop, grouped" view. Collection sits behind the
``ProcessCollector`` protocol so tests can fake it; ``aggregate_processes`` is a
pure function so the grouping/top-N logic is unit-testable on its own.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import psutil
from pydantic import BaseModel

_TOP_GROUPS = 40
_TOP_INSTANCES = 5


class ProcessInstance(BaseModel):
    pid: int
    username: str
    cpu_percent: float
    mem_rss: int
    num_threads: int
    status: str


class ProcessGroup(BaseModel):
    name: str
    count: int
    cpu_percent: float
    mem_rss: int
    instances: list[ProcessInstance]


class ProcessList(BaseModel):
    groups: list[ProcessGroup]
    total: int


def aggregate_processes(
    rows: list[tuple[str, ProcessInstance]],
    *,
    top_groups: int = _TOP_GROUPS,
    top_instances: int = _TOP_INSTANCES,
) -> ProcessList:
    """Group ``(name, instance)`` rows by name and return the top-N groups.

    Grouping happens BEFORE the top-N cut so each group's count and sums cover all
    its processes, not just the ones that survive the cut.
    """
    total = len(rows)
    by_name: dict[str, list[ProcessInstance]] = {}
    for name, inst in rows:
        by_name.setdefault(name, []).append(inst)

    groups: list[ProcessGroup] = []
    for name, insts in by_name.items():
        insts.sort(key=lambda i: i.cpu_percent, reverse=True)
        groups.append(
            ProcessGroup(
                name=name,
                count=len(insts),
                cpu_percent=round(sum(i.cpu_percent for i in insts), 1),
                mem_rss=sum(i.mem_rss for i in insts),
                instances=insts[:top_instances],
            )
        )
    groups.sort(key=lambda g: (g.cpu_percent, g.mem_rss), reverse=True)
    return ProcessList(groups=groups[:top_groups], total=total)


@runtime_checkable
class ProcessCollector(Protocol):
    """The seam the /api/processes endpoint depends on. Tests fake it."""

    def sample(self) -> ProcessList:
        """Return the current processes, aggregated by application."""
        ...


class PsutilProcessCollector:
    """Gather processes via ``psutil.process_iter`` and aggregate them.

    psutil's ``process_iter`` caches Process objects, so ``cpu_percent`` is a
    real non-blocking delta across calls. It is primed in the constructor so the
    first ``sample()`` reports meaningful values.
    """

    _ATTRS = [
        "pid",
        "name",
        "username",
        "cpu_percent",
        "memory_info",
        "num_threads",
        "status",
    ]

    def __init__(self) -> None:
        # Prime per-process cpu_percent (first read is always 0.0).
        for _ in psutil.process_iter(["cpu_percent"]):
            pass

    def sample(self) -> ProcessList:
        rows: list[tuple[str, ProcessInstance]] = []
        for proc in psutil.process_iter(self._ATTRS):
            info: dict[str, Any] = proc.info
            mem = info.get("memory_info")
            name = info.get("name") or f"pid {info['pid']}"
            rows.append(
                (
                    name,
                    ProcessInstance(
                        pid=info["pid"],
                        username=info.get("username") or "",
                        cpu_percent=info.get("cpu_percent") or 0.0,
                        mem_rss=(mem.rss if mem is not None else 0),
                        num_threads=info.get("num_threads") or 0,
                        status=info.get("status") or "",
                    ),
                )
            )
        return aggregate_processes(rows)
