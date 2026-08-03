"""Storage inspection: what is on the disks, and what could be freed.

Filesystem usage comes from psutil (no subprocess, same source as
``metrics.PsutilCollector`` so the dashboard and the agent cannot disagree).
Everything Nix-shaped goes through the runner.

One cost split matters and is deliberate: generations and filesystem usage are
cheap (measured 0.11s and instant on this host) and feed the dashboard's polled
overview, while ``reclaimable_space`` walks the entire store and
``largest_directories`` walks a subtree - both are on-demand only and carry their
own much longer timeouts. Putting either behind a poll would make the live
dashboard hostage to a store walk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import psutil
from pydantic import BaseModel, Field

from .models import Availability, Bounded, Report, clamp
from .run import DEFAULT_TIMEOUT, Runner

# Pseudo-filesystems that answer no storage question anyone asks.
_SKIP_FSTYPES = frozenset(
    {"tmpfs", "devtmpfs", "squashfs", "overlay", "ramfs", "efivarfs", "autofs"}
)

DEFAULT_DIR_LIMIT = 20
MAX_DIR_LIMIT = 100
MAX_DIR_DEPTH = 3
# A store walk is minutes-scale work in the worst case; these are the only two
# inspections allowed past the default timeout.
DIR_SCAN_TIMEOUT = 60.0
GC_DRY_RUN_TIMEOUT = 180.0

NIX_STORE = "/nix/store"

# "7974 store paths would be deleted" - the summary line nix prints. Measured on
# this host: a COUNT and no byte total, which is why the report exposes
# `dead_paths` and NOT a fabricated "space freed".
_DEAD_PATHS = re.compile(r"(\d+)\s+store paths? would be deleted", re.IGNORECASE)


class FilesystemUsage(BaseModel):
    mountpoint: str
    device: str = ""
    fstype: str = ""
    total: int
    used: int
    free: int
    percent: float


class Generation(BaseModel):
    """One NixOS system generation."""

    number: int
    date: str = ""
    nixos_version: str = ""
    kernel_version: str = ""
    configuration_revision: str = ""
    current: bool = False


class GenerationList(Report):
    generations: list[Generation] = Field(default_factory=list)

    @property
    def current(self) -> Generation | None:
        return next((g for g in self.generations if g.current), None)


class FilesystemReport(Report):
    filesystems: list[FilesystemUsage] = Field(default_factory=list)

    @property
    def fullest(self) -> FilesystemUsage | None:
        return max(self.filesystems, key=lambda f: f.percent, default=None)


class DirectoryUsage(BaseModel):
    path: str
    bytes: int


class LargestDirectories(Report, Bounded):
    root: str = ""
    depth: int = 1
    directories: list[DirectoryUsage] = Field(default_factory=list)


class ReclaimableSpace(Report):
    """What a garbage collection would remove.

    ``dead_paths`` is the number nix-collect-garbage actually reports.
    ``bytes_reclaimable`` stays None here on purpose: the dry run prints a COUNT
    and no byte total, and presenting the count as a size would be exactly the
    dressed-up-adjacent-information failure the host spike rejected.
    """

    dead_paths: int | None = None
    bytes_reclaimable: int | None = None


class StorageReport(Report):
    """The composed storage answer to "what filled the disk"."""

    filesystems: FilesystemReport = Field(default_factory=FilesystemReport)
    generations: GenerationList = Field(default_factory=GenerationList)
    nix_store: FilesystemUsage | None = None


def filesystem_usage() -> FilesystemReport:
    """Real filesystems and their usage, from psutil (no subprocess)."""
    report = FilesystemReport()
    try:
        partitions = psutil.disk_partitions(all=False)
    except (OSError, RuntimeError) as exc:
        report.available = Availability.unavailable(
            f"the filesystem table could not be read: {exc}"
        )
        return report
    skipped = 0
    for part in partitions:
        if part.fstype in _SKIP_FSTYPES:
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            # A mount we cannot stat is COUNTED, not silently dropped: "3 mounts
            # could not be read" is the difference between a partial answer and
            # a wrong one.
            skipped += 1
            continue
        report.filesystems.append(
            FilesystemUsage(
                mountpoint=part.mountpoint,
                device=part.device,
                fstype=part.fstype,
                total=usage.total,
                used=usage.used,
                free=usage.free,
                percent=usage.percent,
            )
        )
    report.filesystems.sort(key=lambda f: f.percent, reverse=True)
    if skipped:
        report.available = Availability(
            ok=True, caveat=f"{skipped} mount(s) could not be read and are missing here"
        )
    return report


def store_filesystem(report: FilesystemReport | None = None) -> FilesystemUsage | None:
    """The filesystem holding ``/nix/store``, i.e. what the store competes for.

    The store is usually not its own mount, so this resolves to the longest
    mountpoint that is a prefix of ``/nix/store`` rather than looking for an
    exact match.
    """
    report = report if report is not None else filesystem_usage()
    candidates = [
        fs
        for fs in report.filesystems
        if NIX_STORE == fs.mountpoint
        or NIX_STORE.startswith(fs.mountpoint.rstrip("/") + "/")
    ]
    return max(candidates, key=lambda f: len(f.mountpoint), default=None)


def list_generations(
    runner: Runner, *, timeout: float = DEFAULT_TIMEOUT
) -> GenerationList:
    """System generations from ``nixos-rebuild list-generations --json``.

    These are the rollback targets a later mutating task will activate, so they
    are worth reading even in the read-only phase: the operator sees what a
    rollback WOULD have to choose between.
    """
    result = runner(["nixos-rebuild", "list-generations", "--json"], timeout=timeout)
    report = GenerationList(available=Availability.of(result))
    if not result.ok:
        return report
    try:
        rows = json.loads(result.stdout or "[]")
    except ValueError as exc:
        report.available = Availability.unavailable(
            f"nixos-rebuild returned output this build cannot parse: {exc}"
        )
        return report
    if not isinstance(rows, list):
        report.available = Availability.unavailable(
            "nixos-rebuild returned a non-list where a generation array was expected"
        )
        return report
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            number = int(row.get("generation", 0))
        except (TypeError, ValueError):
            continue
        revision = str(row.get("configurationRevision", ""))
        report.generations.append(
            Generation(
                number=number,
                date=str(row.get("date", "")),
                nixos_version=str(row.get("nixosVersion", "")),
                kernel_version=str(row.get("kernelVersion", "")),
                # nixos-rebuild writes the literal string "Unknown" when the
                # flake was built from a dirty tree; carry it as empty rather
                # than showing "Unknown" as if it were a revision.
                configuration_revision="" if revision == "Unknown" else revision,
                current=bool(row.get("current", False)),
            )
        )
    report.generations.sort(key=lambda g: g.number, reverse=True)
    return report


def storage_report(
    runner: Runner, *, timeout: float = DEFAULT_TIMEOUT
) -> StorageReport:
    """Filesystems, the store's filesystem and the generation list together.

    This is the cheap composite the dashboard overview polls; it deliberately
    excludes the store walk (``reclaimable_space``) and the subtree walk
    (``largest_directories``).
    """
    filesystems = filesystem_usage()
    generations = list_generations(runner, timeout=timeout)
    report = StorageReport(
        filesystems=filesystems,
        generations=generations,
        nix_store=store_filesystem(filesystems),
    )
    if not filesystems.ok and not generations.ok:
        report.available = Availability.unavailable(
            f"{filesystems.available.reason}; {generations.available.reason}"
        )
    elif not filesystems.ok or not generations.ok:
        failed = filesystems if not filesystems.ok else generations
        report.available = Availability(ok=True, caveat=failed.available.reason)
    return report


def largest_directories(
    runner: Runner,
    root: str,
    *,
    depth: int = 1,
    limit: int = DEFAULT_DIR_LIMIT,
    timeout: float = DIR_SCAN_TIMEOUT,
) -> LargestDirectories:
    """The biggest directories under ``root``, one filesystem only.

    ``du -x`` stays on ``root``'s filesystem, so pointing this at ``/`` does not
    wander into ``/nix/store`` or a network mount and never return. Depth is
    capped because each extra level multiplies the walk.
    """
    capped = clamp(limit, default=DEFAULT_DIR_LIMIT, maximum=MAX_DIR_LIMIT)
    capped_depth = clamp(depth, default=1, maximum=MAX_DIR_DEPTH)
    report = LargestDirectories(root=root, depth=capped_depth, limit=capped)
    path = Path(root).expanduser()
    if not path.is_dir():
        report.available = Availability.unavailable(
            f"{root} is not a directory on this host"
        )
        return report
    result = runner(
        [
            "du",
            "-x",  # do not cross filesystem boundaries
            "-b",  # bytes, so the caller formats sizes rather than parsing "4.0K"
            f"--max-depth={capped_depth}",
            # After `--`, so a root like "-foo" is a path and not an option.
            "--",
            str(path),
        ],
        timeout=timeout,
    )
    # du exits non-zero when ANY subdirectory was unreadable, while still
    # printing everything it could read. A partial answer with a caveat beats
    # discarding real data over an unreadable dotfile directory.
    usable = result.ok or bool(result.stdout.strip())
    if not usable:
        report.available = Availability.of(result)
        return report
    if not result.ok:
        report.available = Availability(
            ok=True,
            caveat=(
                "some directories could not be read and are missing from this "
                f"total ({result.reason()})"
            ),
        )
    rows: list[DirectoryUsage] = []
    for line in result.stdout.splitlines():
        size, sep, name = line.partition("\t")
        if not sep:
            continue
        try:
            rows.append(DirectoryUsage(path=name.strip(), bytes=int(size)))
        except ValueError:
            continue
    # du's own total for `root` is the last row and is not a child; drop it so
    # the list is comparable entries rather than one giant and its parts.
    rows = [row for row in rows if row.path != str(path)]
    rows.sort(key=lambda d: d.bytes, reverse=True)
    report.total_seen = len(rows)
    report.directories = rows[:capped]
    report.truncated = len(rows) > capped
    return report


def reclaimable_space(
    runner: Runner,
    *,
    timeout: float = GC_DRY_RUN_TIMEOUT,
) -> ReclaimableSpace:
    """How many store paths a garbage collection would delete.

    Uses ``nix-store --gc --print-dead``, which enumerates the dead set without
    deleting any of it. There is deliberately no ``--delete-older-than`` option
    here even in dry-run form: that flag is documented as equivalent to
    ``nix-env --delete-generations`` on each profile, so it carries a
    generation-deleting half, and its read-only behaviour would then be a
    property of nix's ``--dry-run`` handling rather than of this code. In a
    package whose whole premise is that reading the host cannot change it, the
    guarantee has to come from the command chosen, not from another program's
    promise. Trimming generations by age is a privileged, approved action
    belonging to the host-action framework.

    Two honest caveats about the cost, both measured: the walk takes the global
    GC lock for its duration, so a concurrent ``nix build`` waits on it, and nix
    prunes its own stale temproots files along the way. No store path is
    removed.
    """
    result = runner(["nix-store", "--gc", "--print-dead"], timeout=timeout)
    report = ReclaimableSpace(available=Availability.of(result))
    if not result.ok:
        return report
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    paths = [line for line in lines if line.startswith("/nix/store/")]
    if len(paths) != len(lines):
        # Output this build cannot classify. A count derived from a partly
        # unrecognised listing would be a guess presented as a measurement.
        # (Some nix versions print a summary line instead of a listing; accept
        # that shape explicitly rather than silently miscounting it.)
        match = _DEAD_PATHS.search(result.stdout + result.stderr)
        if match is not None:
            report.dead_paths = int(match.group(1))
            return report
        report.available = Availability(
            ok=True,
            caveat=(
                "nix-store printed output this build cannot read as a dead-path "
                "listing, so the count is unknown rather than zero"
            ),
        )
        return report
    # An EMPTY listing means zero dead paths - a real answer, and the common one
    # on a freshly collected store. Measured on this host, `--print-dead` writes
    # bare store paths and no summary line at all, so treating "no output" as
    # unrecognised would report "unknown" for the healthiest possible result -
    # the exact empty-vs-broken confusion this package exists to prevent.
    report.dead_paths = len(paths)
    return report
