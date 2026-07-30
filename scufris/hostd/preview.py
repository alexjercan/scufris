"""What the operator is shown before approving, and how it can be undone.

The rule this module exists to enforce, from the spike: **where a class has no
honest preview, the preview says so rather than presenting adjacent information
as one.** Only R2 and R3 can be simulated. R1 cannot - a restart is not a dry
run - so its preview is explicitly labelled a statement of current state and
blast radius, and never rendered as a prediction.

The second rule, also paid for once already: an empty preview and a broken one
must not look the same. Every preview carries an ``Availability``, so "asked,
answered, nothing to show" is a different sentence from "the preview failed".

R1 and R2 live here. R3 lives in ``nixos.py``, which also documents the one thing
a configuration preview deliberately does NOT include - the unit-restart list,
because producing it means running an unapproved configuration's own code as
root.
"""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, Field

from ..host.models import Availability
from ..host.run import Runner, nix_cli
from ..host.storage import list_generations
from ..host.units import Scope, unit_status
from .actions import (
    GENERATION_TIMEOUT,
    PROTECTED_GENERATIONS,
    R3_KINDS,
    UNIT_KINDS,
    ActionKind,
    Plan,
)
from .files import DEFAULT_FILES, Files

# Enumerating the dead set walks the whole store and takes the GC lock; measured
# on this host it ran for 35 seconds. A preview of a slow operation is allowed to
# be slow, but it is still bounded.
DEAD_SET_TIMEOUT = 300.0
PATH_INFO_TIMEOUT = 120.0

# How many store paths one `nix path-info --json` call is given. Measured: 500
# paths answer in ~0.2s, and the argv stays far inside ARG_MAX.
_PATH_INFO_BATCH = 500

# Past this many dead paths the size is reported as uncomputed rather than
# spending minutes on a number nobody waited for. Stated explicitly - an unknown
# size must not render as zero.
_MAX_SIZED_PATHS = 40000

# How many reverse dependencies are listed before the rest are summarised.
_MAX_REVERSE_DEPS = 25


class PreviewKind(StrEnum):
    """How much the preview actually knows.

    The distinction is load-bearing for the approval surface: a SIMULATION is
    the system's own answer to "what would happen", while STATE is a description
    of the world as it is now, from which the operator infers the consequence.
    Rendering the second as the first is the failure this taxonomy prevents.
    """

    SIMULATION = "simulation"
    STATE = "state"
    NONE = "none"


class Reversal(BaseModel):
    """How an applied action can be undone, or why it cannot be.

    ``possible=False`` is a first-class answer, not a missing value. R2 is
    one-way by nature, and a restart cannot be un-restarted; both must be said
    plainly at approval time rather than discovered afterwards.
    """

    possible: bool
    summary: str
    kind: ActionKind | None = None
    args: dict[str, object] = Field(default_factory=dict)


class Preview(BaseModel):
    """Everything shown to the operator about a proposed action."""

    kind: PreviewKind
    headline: str
    # The honesty label: what these lines ARE. Rendered next to them, always.
    label: str
    lines: list[str] = Field(default_factory=list)
    available: Availability = Field(default_factory=Availability)

    @property
    def ok(self) -> bool:
        return self.available.ok


class Fingerprint(BaseModel):
    """The state a preview was taken against.

    Re-read at apply time. If it moved, the operator approved a description of a
    system that no longer exists, and the proposal is refused rather than
    applied against the new one.
    """

    value: str
    describes: str

    def matches(self, other: "Fingerprint") -> bool:
        return self.value == other.value


def _human_bytes(count: int) -> str:
    """Render a byte count the way an operator reads it."""
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TiB"


def dead_set_size(runner: Runner) -> tuple[int | None, int | None, str]:
    """The store paths that are already unreachable, and what they weigh.

    Returns ``(path_count, total_bytes, caveat)``. ``total_bytes`` is None when
    the size was not computed, and the caveat says why - never zero.

    The size is the sum of each path's own ``narSize``, deliberately NOT
    ``nix path-info -S`` (closure size). Closures overlap, so summing them over
    a set of paths counts shared dependencies once per referrer and produces a
    number several times larger than the space that would actually be freed.
    The spike named ``-S``; measuring it showed the sum would be a fiction, and
    a fiction in the "space freed" field is exactly the failure this framework
    exists to refuse.
    """
    listing = runner(["nix-store", "--gc", "--print-dead"], timeout=DEAD_SET_TIMEOUT)
    if not listing.ok:
        return None, None, listing.reason()
    paths = [
        line.strip()
        for line in listing.stdout.splitlines()
        if line.strip().startswith("/nix/store/")
    ]
    if not paths:
        # A real answer, and the healthiest one: nothing is dead.
        return 0, 0, ""
    if len(paths) > _MAX_SIZED_PATHS:
        return (
            len(paths),
            None,
            f"{len(paths)} dead paths is past the {_MAX_SIZED_PATHS} this "
            "preview will size, so the total is unknown rather than zero",
        )
    total = 0
    for start in range(0, len(paths), _PATH_INFO_BATCH):
        batch = paths[start : start + _PATH_INFO_BATCH]
        result = runner(
            nix_cli("path-info", "--json", "--", *batch), timeout=PATH_INFO_TIMEOUT
        )
        if not result.ok:
            return len(paths), None, f"path sizes are unavailable: {result.reason()}"
        try:
            info = json.loads(result.stdout or "{}")
        except ValueError as exc:
            return (
                len(paths),
                None,
                f"nix printed path info this build cannot read: {exc}",
            )
        if not isinstance(info, dict):
            return len(paths), None, "nix returned path info in an unexpected shape"
        for entry in info.values():
            if isinstance(entry, dict):
                size = entry.get("narSize")
                if isinstance(size, int):
                    total += size
    return len(paths), total, ""


def _reverse_dependencies(runner: Runner, unit: str) -> tuple[list[str], str]:
    """Units that depend on ``unit``, i.e. what else this action reaches.

    Returns the (bounded) list and a caveat when it could not be read. This is
    the blast radius half of an R1 preview.
    """
    result = runner(
        [
            "systemctl",
            "list-dependencies",
            "--reverse",
            "--plain",
            "--no-pager",
            "--",
            unit,
        ],
        timeout=30.0,
    )
    if not result.ok:
        return [], result.reason()
    names = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and line.strip() != unit
    ]
    return names, ""


_EXPECTED_STATE: dict[ActionKind, str] = {
    ActionKind.UNIT_START: "active",
    ActionKind.UNIT_STOP: "inactive",
    ActionKind.UNIT_RESTART: "active",
    ActionKind.UNIT_RELOAD: "active",
}


def _unit_preview(plan: Plan, runner: Runner) -> tuple[Preview, Fingerprint, Reversal]:
    unit = str(plan.args["unit"])
    status = unit_status(runner, unit, scope=Scope.SYSTEM)
    if not status.ok:
        unavailable = Preview(
            kind=PreviewKind.NONE,
            headline=f"{plan.summary}",
            label="the unit's current state could not be read",
            available=Availability.unavailable(status.available.reason),
        )
        return (
            unavailable,
            Fingerprint(value="", describes="unreadable"),
            Reversal(
                possible=False,
                summary=(
                    "the prior state could not be recorded, so no inverse can be "
                    "offered"
                ),
            ),
        )
    if not status.loaded:
        unavailable = Preview(
            kind=PreviewKind.NONE,
            headline=plan.summary,
            label="systemd has no such unit",
            available=Availability.unavailable(
                f"{unit} is not a unit systemd knows about (LoadState="
                f"{status.load_state or 'unknown'})"
            ),
        )
        return (
            unavailable,
            Fingerprint(value="", describes="not-found"),
            Reversal(possible=False, summary="the unit does not exist"),
        )

    deps, deps_caveat = _reverse_dependencies(runner, unit)
    shown = deps[:_MAX_REVERSE_DEPS]
    lines = [
        f"unit: {unit} ({status.description})"
        if status.description
        else f"unit: {unit}",
        f"now:  {status.active_state} ({status.sub_state}), "
        f"unit file {status.unit_file_state or 'unknown'}",
        f"after: {_EXPECTED_STATE[plan.kind]}",
        f"command: {' '.join(plan.steps[0].argv)}",
    ]
    if deps_caveat:
        lines.append(f"reverse dependencies: unreadable ({deps_caveat})")
    elif not deps:
        lines.append("reverse dependencies: none - nothing else declares it needs this")
    else:
        lines.append(f"reverse dependencies ({len(deps)}), which this action reaches:")
        lines.extend(f"  - {name}" for name in shown)
        if len(deps) > len(shown):
            lines.append(f"  ... and {len(deps) - len(shown)} more")

    preview = Preview(
        kind=PreviewKind.STATE,
        headline=plan.summary,
        label=(
            "systemd cannot simulate this. These lines are the unit's CURRENT "
            "state and what else depends on it - a statement of blast radius, "
            "not a prediction of the outcome."
        ),
        lines=lines,
        available=Availability(caveat=deps_caveat) if deps_caveat else Availability(),
    )
    fingerprint = Fingerprint(
        value=f"{status.load_state}:{status.active_state}:{status.sub_state}",
        describes=f"{unit} load/active/sub state",
    )
    return preview, fingerprint, _unit_reversal(plan, status.active_state)


def _unit_reversal(plan: Plan, active_state: str) -> Reversal:
    """The inverse of an R1 action, given the state it is being applied to.

    A restart of an already-running daemon is the honest ``possible=False``
    case: the unit ends where it started, so there is no state to restore, and
    the thing that actually changed - the process and everything it held in
    memory - cannot be put back.
    """
    unit = str(plan.args["unit"])
    was_active = active_state == "active"
    if plan.kind is ActionKind.UNIT_START:
        if was_active:
            return Reversal(
                possible=False,
                summary=f"{unit} is already active; starting it changes nothing to undo",
            )
        return Reversal(
            possible=True,
            summary=f"stop {unit}, restoring it to inactive",
            kind=ActionKind.UNIT_STOP,
            args={"unit": unit},
        )
    if plan.kind is ActionKind.UNIT_STOP:
        if not was_active:
            return Reversal(
                possible=False,
                summary=f"{unit} is already {active_state or 'inactive'}; "
                "stopping it changes nothing to undo",
            )
        return Reversal(
            possible=True,
            summary=(
                f"start {unit}, restoring it to active - though a stopped daemon "
                "has lost whatever it held in memory"
            ),
            kind=ActionKind.UNIT_START,
            args={"unit": unit},
        )
    # restart / reload
    if was_active:
        return Reversal(
            possible=False,
            summary=(
                f"{unit} ends active, where it already is, so there is no state "
                "to restore - and the process it was running is gone along with "
                "everything it held in memory. This cannot be undone."
            ),
        )
    return Reversal(
        possible=True,
        summary=f"stop {unit}, restoring it to {active_state or 'inactive'}",
        kind=ActionKind.UNIT_STOP,
        args={"unit": unit},
    )


_GC_REVERSAL = Reversal(
    possible=False,
    summary=(
        "ONE-WAY. Deleted store paths cannot be restored - they have to be "
        "rebuilt or refetched - and a deleted generation cannot be recovered. "
        "There is no undo for this action."
    ),
)


def _gc_store_preview(
    plan: Plan, runner: Runner
) -> tuple[Preview, Fingerprint, Reversal]:
    count, total, caveat = dead_set_size(runner)
    if count is None:
        return (
            Preview(
                kind=PreviewKind.NONE,
                headline=plan.summary,
                label="the dead set could not be enumerated",
                available=Availability.unavailable(caveat),
            ),
            Fingerprint(value="", describes="unreadable"),
            _GC_REVERSAL,
        )
    lines = [
        f"store paths already unreachable: {count}",
        (
            f"space this would free: {_human_bytes(total)}"
            if total is not None
            else "space this would free: not computed"
        ),
        f"command: {' '.join(plan.steps[0].argv)}",
        "no system generation is touched by this verb",
    ]
    if count == 0:
        lines.append("nothing is dead right now, so this would delete nothing")
    return (
        Preview(
            kind=PreviewKind.SIMULATION,
            headline=plan.summary,
            label=(
                "counted from nix's own dead-path enumeration. The size is the "
                "sum of each path's nar size, so it is the space actually freed, "
                "not a sum of overlapping closures."
            ),
            lines=lines,
            available=Availability(caveat=caveat) if caveat else Availability(),
        ),
        Fingerprint(value=f"dead:{count}", describes="the number of dead store paths"),
        _GC_REVERSAL,
    )


def _gc_older_than_preview(
    plan: Plan, runner: Runner
) -> tuple[Preview, Fingerprint, Reversal]:
    days = int(str(plan.args["days"]))
    listing = list_generations(runner, timeout=GENERATION_TIMEOUT)
    if not listing.ok:
        return (
            Preview(
                kind=PreviewKind.NONE,
                headline=plan.summary,
                label="the generation list could not be read",
                available=Availability.unavailable(listing.available.reason),
            ),
            Fingerprint(value="", describes="unreadable"),
            _GC_REVERSAL,
        )
    by_number = {g.number: g for g in listing.generations}
    removed = [by_number[n] for n in plan.generations_removed if n in by_number]
    kept = [g for g in listing.generations if g.number not in plan.generations_removed]

    # The listed generations ARE the argv (actions.build_plan names them), so
    # this preview and the command cannot disagree - which is the whole point of
    # naming them rather than passing an age flag.
    lines = [f"command: {' '.join(plan.steps[0].argv)}"]
    lines.append(f"generations this deletes ({len(removed)}), older than {days} days:")
    lines.extend(f"  - {g.number} ({g.date}) {g.nixos_version}" for g in removed)
    lines.append(f"generations kept ({len(kept)}):")
    lines.extend(f"  - {g.number} ({g.date})" for g in kept[:_MAX_REVERSE_DEPS])
    if len(kept) > _MAX_REVERSE_DEPS:
        lines.append(f"  ... and {len(kept) - len(kept[:_MAX_REVERSE_DEPS])} more")
    lines.append(
        f"the {PROTECTED_GENERATIONS} most recent are excluded by position "
        "before age is considered, so the current system and its rollback "
        "target are never in this list"
    )
    lines.append(
        "this frees no disk space by itself: it removes the generation links, "
        "which makes the store paths they were holding collectable. A separate "
        "gc_store action reclaims them, with its own preview and its own "
        "approval."
    )

    return (
        Preview(
            kind=PreviewKind.SIMULATION,
            headline=plan.summary,
            label=(
                "the exact generations this command names, resolved by the "
                "helper. No size is shown because this action frees no space on "
                "its own - a figure here would describe a different action."
            ),
            lines=lines,
        ),
        Fingerprint(
            value="gen:" + ",".join(str(g.number) for g in listing.generations),
            describes="the system generation list",
        ),
        _GC_REVERSAL,
    )


def build_preview(
    plan: Plan, runner: Runner, files: Files = DEFAULT_FILES
) -> tuple[Preview, Fingerprint, Reversal]:
    """Preview ``plan``, fingerprint the state it was taken against, and say
    how it can be undone."""
    if plan.kind in UNIT_KINDS:
        return _unit_preview(plan, runner)
    if plan.kind is ActionKind.GC_STORE:
        return _gc_store_preview(plan, runner)
    if plan.kind in R3_KINDS:
        # Imported here rather than at module scope: `nixos` needs this module's
        # Preview/Fingerprint/Reversal models, so a top-level import would be a
        # cycle. The dispatch belongs with the other classes, so the import moves
        # instead of the dispatch.
        from . import nixos

        if plan.kind is ActionKind.ACTIVATE:
            return nixos.activate_preview(plan, runner, files)
        return nixos.rollback_preview(plan, runner, files)
    return _gc_older_than_preview(plan, runner)


def read_fingerprint(
    plan: Plan, runner: Runner, files: Files = DEFAULT_FILES
) -> Fingerprint:
    """Re-read the fingerprint at apply time, without rebuilding the preview.

    Cheaper than a full preview for the R2 and R3 verbs (no dead-set walk, no
    closure diff), and it is the only thing apply actually needs: did the world
    move.
    """
    if plan.kind in R3_KINDS:
        from . import nixos

        return nixos.r3_fingerprint(runner, files)
    if plan.kind in UNIT_KINDS:
        unit = str(plan.args["unit"])
        status = unit_status(runner, unit, scope=Scope.SYSTEM)
        if not status.ok or not status.loaded:
            return Fingerprint(value="", describes="unreadable")
        return Fingerprint(
            value=f"{status.load_state}:{status.active_state}:{status.sub_state}",
            describes=f"{unit} load/active/sub state",
        )
    if plan.kind is ActionKind.GC_STORE:
        count, _total, caveat = dead_set_size(runner)
        if count is None:
            return Fingerprint(value="", describes=caveat or "unreadable")
        return Fingerprint(
            value=f"dead:{count}", describes="the number of dead store paths"
        )
    listing = list_generations(runner, timeout=GENERATION_TIMEOUT)
    if not listing.ok:
        return Fingerprint(value="", describes="unreadable")
    return Fingerprint(
        value="gen:" + ",".join(str(g.number) for g in listing.generations),
        describes="the system generation list",
    )
