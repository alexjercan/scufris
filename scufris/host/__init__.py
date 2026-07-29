"""Read-only inspection of the host scufris runs on.

The existing ``metrics``/``processes`` collectors answer "what is using CPU right
now". This package answers the rest of the questions an operator asks about a
NixOS box: what failed, what the logs say, what filled the disk, what is
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

``HostInspector`` is the facade: it holds the runner and the host-specific paths
so the dashboard endpoint and the MCP tools construct one object instead of
threading four arguments through every call.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .journal import (
    DEFAULT_JOURNAL_LINES,
    MAX_JOURNAL_BYTES,
    MAX_JOURNAL_LINES,
    JournalEntry,
    JournalReport,
    read_journal,
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
from .run import (
    DEFAULT_TIMEOUT,
    CommandResult,
    FakeRunner,
    Outcome,
    Runner,
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
    "CommandResult",
    "DEFAULT_JOURNAL_LINES",
    "DEFAULT_TIMEOUT",
    "DEFAULT_UNIT_LIMIT",
    "FakeRunner",
    "FilesystemReport",
    "FilesystemUsage",
    "FirewallReport",
    "FlakeStatus",
    "Generation",
    "GenerationList",
    "HostInspector",
    "HostOverview",
    "InterfaceReport",
    "JournalEntry",
    "JournalReport",
    "LargestDirectories",
    "ListeningSocket",
    "MAX_JOURNAL_BYTES",
    "MAX_JOURNAL_LINES",
    "MAX_UNIT_LIMIT",
    "NetworkReport",
    "Outcome",
    "ProfileReport",
    "ReclaimableSpace",
    "Report",
    "Runner",
    "Scope",
    "SocketReport",
    "StorageReport",
    "ThermalReport",
    "ThrottleCounters",
    "UnitList",
    "UnitStatus",
    "UnitSummary",
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
    "ok_result",
    "profile_contents",
    "read_journal",
    "reclaimable_space",
    "run_command",
    "storage_report",
    "thermal_report",
    "unit_status",
    "what_provides",
]

# Where the operator's NixOS flake lives. Only read here (flake.lock); the
# mutating tasks will write to a sprout worktree over it, never in place.
DEFAULT_CONFIG_REPO = Path.home() / "personal" / "nix.dotfiles"


class HostOverview(BaseModel):
    """The cheap, glanceable subset the dashboard polls.

    Every member here was MEASURED cheap on this host (failed units 0.00s,
    generations 0.11s, filesystems in-process). The expensive inspections -
    ``reclaimable_space`` walks the whole store, ``largest_directories`` walks a
    subtree - are deliberately absent: putting either behind a poll would make
    the live dashboard hostage to a store walk.
    """

    failed_system_units: UnitList = Field(default_factory=UnitList)
    failed_user_units: UnitList = Field(default_factory=UnitList)
    storage: StorageReport = Field(default_factory=StorageReport)
    thermal: ThermalReport = Field(default_factory=ThermalReport)


class HostInspector:
    """Facade over the inspection modules, holding the runner and host paths.

    Constructed once per process (the dashboard) or per tool call (the MCP
    server). Stateless apart from its configuration, so it is safe to share.
    """

    def __init__(
        self,
        runner: Runner = run_command,
        *,
        config_repo: Path | None = None,
        system: Path = CURRENT_SYSTEM,
        cpu_sysfs: Path = CPU_SYSFS,
    ) -> None:
        self._runner = runner
        # expanduser at USE time, not at config-read time: pydantic stores a "~"
        # env value verbatim.
        self._config_repo = (config_repo or DEFAULT_CONFIG_REPO).expanduser()
        self._system = system
        self._cpu_sysfs = cpu_sysfs

    # --- systemd ---------------------------------------------------------

    def list_units(
        self,
        *,
        scope: Scope = Scope.SYSTEM,
        state: str = "",
        pattern: str = "",
        limit: int = DEFAULT_UNIT_LIMIT,
    ) -> UnitList:
        return list_units(
            self._runner, scope=scope, state=state, pattern=pattern, limit=limit
        )

    def failed_units(self, *, scope: Scope = Scope.SYSTEM) -> UnitList:
        return failed_units(self._runner, scope=scope)

    def unit_status(self, name: str, *, scope: Scope = Scope.SYSTEM) -> UnitStatus:
        return unit_status(self._runner, name, scope=scope)

    # --- journal ---------------------------------------------------------

    def journal(
        self,
        *,
        unit: str = "",
        scope: Scope = Scope.SYSTEM,
        priority: str = "",
        since: str = "",
        until: str = "",
        lines: int = DEFAULT_JOURNAL_LINES,
    ) -> JournalReport:
        return read_journal(
            self._runner,
            unit=unit,
            scope=scope,
            priority=priority,
            since=since,
            until=until,
            lines=lines,
        )

    # --- storage ---------------------------------------------------------

    def storage(self) -> StorageReport:
        return storage_report(self._runner)

    def generations(self) -> GenerationList:
        return list_generations(self._runner)

    def largest_directories(
        self, root: str, *, depth: int = 1, limit: int = 20
    ) -> LargestDirectories:
        return largest_directories(self._runner, root, depth=depth, limit=limit)

    def reclaimable_space(self) -> ReclaimableSpace:
        return reclaimable_space(self._runner)

    # --- network ---------------------------------------------------------

    def network(self) -> NetworkReport:
        return network_report(self._runner, system=self._system)

    def firewall(self) -> FirewallReport:
        return declared_firewall(self._system)

    # --- thermal ---------------------------------------------------------

    def thermal(self) -> ThermalReport:
        return thermal_report(self._cpu_sysfs)

    # --- packages --------------------------------------------------------

    def what_provides(self, binary: str) -> BinaryProvider:
        return what_provides(binary)

    def profile(self, *, limit: int = 40) -> ProfileReport:
        return profile_contents(limit=limit)

    def closure_diff(self, before: str | int, after: str | int) -> ClosureDiff:
        return closure_diff(self._runner, before, after)

    def flake_status(self) -> FlakeStatus:
        return flake_status(self._config_repo)

    # --- composed --------------------------------------------------------

    def overview(self) -> HostOverview:
        """The dashboard's snapshot: cheap inspections only."""
        return HostOverview(
            failed_system_units=self.failed_units(scope=Scope.SYSTEM),
            failed_user_units=self.failed_units(scope=Scope.USER),
            storage=self.storage(),
            thermal=self.thermal(),
        )
