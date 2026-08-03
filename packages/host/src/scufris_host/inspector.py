"""The inspection facade: one object holding the runner and the host's paths.

Its own module rather than a piece of ``__init__`` so the package's door stays a
door. ``overview.py`` needs :class:`HostInspector` and :class:`HostOverview` at
import time; if they lived in ``__init__`` the facade could not re-export
``overview``'s names without a cycle, and every later module that wants the
inspector would inherit the same problem.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .journal import DEFAULT_JOURNAL_LINES, JournalReport, read_journal
from .network import (
    CURRENT_SYSTEM,
    FirewallReport,
    NetworkReport,
    declared_firewall,
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
from .run import Runner, run_command
from .storage import (
    GenerationList,
    LargestDirectories,
    ReclaimableSpace,
    StorageReport,
    largest_directories,
    list_generations,
    reclaimable_space,
    storage_report,
)
from .thermal import CPU_SYSFS, ThermalReport, thermal_report
from .units import (
    DEFAULT_UNIT_LIMIT,
    Scope,
    UnitList,
    UnitStatus,
    failed_units,
    list_units,
    unit_status,
)

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
