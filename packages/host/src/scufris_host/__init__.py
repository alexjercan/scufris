"""Read-only inspection of the host scufris runs on.

The ``metrics``/``processes`` collectors answer "what is using CPU right now".
The rest of this distribution answers the other questions an operator asks about
a NixOS box: what failed, what the logs say, what filled the disk, what is
listening, whether the CPU throttled, what provides a binary, and how the current
generation differs from the previous one.

Three rules hold across every module here, and they are the reason it is a
package rather than a pile of shell calls:

1. **One door to the outside.** Every command goes through the ``Runner`` seam in
   ``run.py`` and comes back as a classified ``CommandResult``. Nothing calls
   ``subprocess`` directly, so a missing binary, a denied permission and a
   timeout are three different sentences rather than three different tracebacks.
2. **Availability lives on the model.** Every report embeds an ``Availability``.
   A tool never raises at the MCP boundary, and an empty result is never
   confusable with a broken one - the distinction the whole package exists to
   preserve.
3. **Everything is bounded.** Journal reads, unit listings, directory walks and
   closure diffs all carry caps, and a capped result says so.

``HostInspector`` holds the runner and the host-specific paths so the dashboard
endpoint and the MCP tools construct one object instead of threading four
arguments through every call.

THIS MODULE IS THE ONLY DOOR. Every name a consumer needs is re-exported here in
one flat namespace, and ``test_no_package_imports_a_sibling_private_module``
holds every sibling distribution to it: ``from scufris_host import Runner``,
never ``from scufris_host.run import Runner``. Flat rather than a set of
addressable submodules because the three name sets it merges - the reports',
``metrics``' and ``processes``' - are pairwise disjoint, so flattening costs no
aliasing. A name later added to any of them must not collide with one already
here; the single ``__all__`` makes a collision a visible edit rather than a
silent shadow.

``render`` is the one name imported as a MODULE (``from scufris_host import
render``, by the MCP tools). That is still a facade import - it names the
distribution root and nothing inside it - and it keeps fourteen renderers out of
the flat namespace instead of swallowing them into it.
"""

from __future__ import annotations

from .inspector import HostInspector, HostOverview
from .journal import (
    DEFAULT_JOURNAL_LINES,
    MAX_JOURNAL_BYTES,
    MAX_JOURNAL_LINES,
    JournalEntry,
    JournalReport,
    read_journal,
)
from .metrics import (
    Collector,
    CpuActivity,
    DiskIoRate,
    DiskUsage,
    FanReading,
    GpuRunner,
    GpuStats,
    HostStats,
    MemStats,
    NetIfRate,
    NetIO,
    PsutilCollector,
    SensorGroup,
    SensorReading,
    SwapStats,
    parse_gpus,
)
from .models import Availability, Bounded, Report, clamp
from .network import (
    CURRENT_SYSTEM,
    FirewallReport,
    InterfaceReport,
    ListeningSocket,
    NetworkReport,
    SocketReport,
    declared_firewall,
    list_interfaces,
    listening_sockets,
    network_report,
)
from .overview import MIN_HOST_OVERVIEW_TTL, HostOverviewCache
from .packages import (
    BinaryProvider,
    ClosureDiff,
    FlakeStatus,
    ProfileReport,
    closure_diff,
    flake_status,
    profile_contents,
    what_provides,
)
from .processes import (
    ProcessCollector,
    ProcessGroup,
    ProcessInstance,
    ProcessList,
    PsutilProcessCollector,
    aggregate_processes,
)
from .run import (
    DEFAULT_TIMEOUT,
    NIX_FEATURES,
    CommandResult,
    FakeRunner,
    Outcome,
    Runner,
    nix_cli,
    ok_result,
    run_command,
)
from .storage import (
    FilesystemReport,
    FilesystemUsage,
    Generation,
    GenerationList,
    LargestDirectories,
    ReclaimableSpace,
    StorageReport,
    filesystem_usage,
    largest_directories,
    list_generations,
    reclaimable_space,
    storage_report,
)
from .thermal import (
    CPU_SYSFS,
    BatteryState,
    ThermalReport,
    ThrottleCounters,
    thermal_report,
)
from .units import (
    DEFAULT_UNIT_LIMIT,
    MAX_UNIT_LIMIT,
    Scope,
    UnitList,
    UnitStatus,
    UnitSummary,
    failed_units,
    list_units,
    unit_status,
)

__all__ = [
    "Availability",
    "BatteryState",
    "BinaryProvider",
    "Bounded",
    "CPU_SYSFS",
    "CURRENT_SYSTEM",
    "ClosureDiff",
    "Collector",
    "CommandResult",
    "CpuActivity",
    "DEFAULT_JOURNAL_LINES",
    "DEFAULT_TIMEOUT",
    "DEFAULT_UNIT_LIMIT",
    "DiskIoRate",
    "DiskUsage",
    "FakeRunner",
    "FanReading",
    "FilesystemReport",
    "FilesystemUsage",
    "FirewallReport",
    "FlakeStatus",
    "Generation",
    "GenerationList",
    "GpuRunner",
    "GpuStats",
    "HostInspector",
    "HostOverview",
    "HostOverviewCache",
    "HostStats",
    "InterfaceReport",
    "JournalEntry",
    "JournalReport",
    "LargestDirectories",
    "ListeningSocket",
    "MAX_JOURNAL_BYTES",
    "MAX_JOURNAL_LINES",
    "MAX_UNIT_LIMIT",
    "MIN_HOST_OVERVIEW_TTL",
    "MemStats",
    "NIX_FEATURES",
    "NetIO",
    "NetIfRate",
    "NetworkReport",
    "Outcome",
    "ProcessCollector",
    "ProcessGroup",
    "ProcessInstance",
    "ProcessList",
    "ProfileReport",
    "PsutilCollector",
    "PsutilProcessCollector",
    "ReclaimableSpace",
    "Report",
    "Runner",
    "Scope",
    "SensorGroup",
    "SensorReading",
    "SocketReport",
    "StorageReport",
    "SwapStats",
    "ThermalReport",
    "ThrottleCounters",
    "UnitList",
    "UnitStatus",
    "UnitSummary",
    "aggregate_processes",
    "clamp",
    "closure_diff",
    "declared_firewall",
    "failed_units",
    "filesystem_usage",
    "flake_status",
    "largest_directories",
    "list_generations",
    "list_interfaces",
    "list_units",
    "listening_sockets",
    "network_report",
    "nix_cli",
    "ok_result",
    "parse_gpus",
    "profile_contents",
    "read_journal",
    "reclaimable_space",
    "run_command",
    "storage_report",
    "thermal_report",
    "unit_status",
    "what_provides",
]
