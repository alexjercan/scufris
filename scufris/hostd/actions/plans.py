"""Turning a validated verb into the exact commands that would run.

``build_plan`` is the only entry point. It builds every argv itself, from the
verb and the validated arguments - nothing beyond it may add, reorder or
reinterpret an argv element.
"""

from __future__ import annotations

from datetime import datetime

from ...host.run import Runner, nix_cli
from ...host.storage import Generation, list_generations
from ..files import DEFAULT_FILES, Files
from .models import (
    ActivateArgs,
    GcOlderThanArgs,
    Plan,
    RollbackArgs,
    Step,
    UnitArgs,
    parse_args,
)
from .taxonomy import RISK_OF, UNIT_KINDS, ActionKind, ActionRefused, RiskClass
from .validate import normalise_unit, validate_provenance, validate_toplevel

# How many of the most recent generations may never be garbage collected. Two,
# not one: the current generation is what is running, and the one before it is
# the rollback target the R3 class depends on. Measured in the spike,
# `--delete-older-than` keeps only the CURRENT generation and is otherwise
# purely age-based, so this floor is enforced here rather than delegated to the
# flag.
PROTECTED_GENERATIONS = 2

# The NixOS system profile, whose generations are the rollback targets.
SYSTEM_PROFILE = "/nix/var/nix/profiles/system"

# The wall clock a generation listing gets. Short: it is a JSON read.
GENERATION_TIMEOUT = 20.0

# The transient unit the switch runs in. Deliberately the SAME name
# `nixos-rebuild` uses (`SWITCH_TO_CONFIGURATION_CMD_PREFIX` in
# nixos_rebuild/nix.py, read on this host): systemd refuses a second
# `systemd-run --unit=<name>` while one is live, so a hand-run `nixos-rebuild
# switch` and this helper cannot activate two configurations at once. The
# collision is a feature, and `nixos.switch_in_flight` turns it into a sentence
# before the profile is touched rather than a cryptic systemd error after.
SWITCH_UNIT = "nixos-rebuild-switch-to-configuration"

# Pointing the profile at a path is a symlink swap; activation restarts changed
# units and updates the boot entries.
PROFILE_TIMEOUT = 120.0
SWITCH_TIMEOUT = 1800.0

_SYSTEMCTL_VERB: dict[ActionKind, str] = {
    ActionKind.UNIT_START: "start",
    ActionKind.UNIT_STOP: "stop",
    ActionKind.UNIT_RESTART: "restart",
    ActionKind.UNIT_RELOAD: "reload",
}


def _switch_step(toplevel: str) -> Step:
    """The activation itself, run in a transient unit rather than as our child.

    ``systemd-run`` is what ``nixos-rebuild`` does, for a reason that applies
    here with more force: a configuration change can restart the very units this
    helper and scufris run in. As a direct child, the switch would be killed
    halfway through by the thing it was switching - leaving a machine whose
    profile and running state disagree and whose audit record stops mid-sentence.
    In a transient unit it survives that, and it survives this helper being
    restarted.

    The cost is recorded honestly rather than hidden: the transient unit is NOT
    in our process group, so a cancellation cannot stop it (see
    ``Plan.cancel_detail``). Nothing can safely stop a switch halfway anyway.
    """
    return Step(
        argv=[
            "systemd-run",
            "--collect",
            "--no-ask-password",
            "--pipe",
            "--quiet",
            "--service-type=exec",
            f"--unit={SWITCH_UNIT}",
            # An explicit 0 rather than inheriting: a bootloader REINSTALL is a
            # different, riskier act than a switch, and it is not what was
            # previewed.
            "--setenv=NIXOS_INSTALL_BOOTLOADER=0",
            "--",
            f"{toplevel}/bin/switch-to-configuration",
            "switch",
        ],
        label="activate it (restart changed units, update the boot entries)",
        timeout=SWITCH_TIMEOUT,
    )


_R3_CANCEL_DETAIL = (
    f"the switch runs in the transient systemd unit {SWITCH_UNIT}, which is not "
    "in this helper's process group: cancelling stops WATCHING it, not the "
    "activation itself. Read the generation list to see what the switch ended up "
    "doing; nothing can safely stop a switch-to-configuration halfway."
)

_R3_PARTIAL_DETAIL = (
    "the system profile was already pointed at the new configuration when the "
    "switch failed, so THIS boot still runs the old one while the NEXT boot "
    "would run the new one. That is a split state: roll back to the previous "
    "generation, or fix the activation failure and switch again, before "
    "rebooting."
)


def _parse_generation_date(raw: str) -> datetime | None:
    """Parse the ``nixos-rebuild list-generations`` date, or give up honestly."""
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def generations_older_than(
    generations: list[Generation], days: int, *, now: datetime
) -> list[Generation]:
    """The generations an age-based collection would remove, with the floor applied.

    The floor is the whole reason this function exists rather than a flag: the
    two most recent generations are excluded by POSITION before age is even
    considered, so a box that has not been rebuilt in a year cannot have its own
    rollback target collected. A generation whose date cannot be parsed is kept,
    because "we could not tell how old it is" must never resolve to "delete it".
    """
    ordered = sorted(generations, key=lambda g: g.number, reverse=True)
    candidates = ordered[PROTECTED_GENERATIONS:]
    cutoff_seconds = days * 86400.0
    removed: list[Generation] = []
    for generation in candidates:
        when = _parse_generation_date(generation.date)
        if when is None:
            continue
        if (now - when).total_seconds() > cutoff_seconds:
            removed.append(generation)
    return removed


def generation_link(number: int) -> str:
    """The profile link a generation number names."""
    return f"{SYSTEM_PROFILE}-{number}-link"


def _activate_plan(args: ActivateArgs, *, runner: Runner, files: Files) -> Plan:
    toplevel = validate_toplevel(args.toplevel, runner=runner, files=files)
    repo, rev = validate_provenance(args.repo, args.rev)
    provenance = f" built from {rev[:12]}" if rev else ""
    return Plan(
        kind=ActionKind.ACTIVATE,
        risk=RiskClass.R3,
        args={"toplevel": toplevel, "repo": repo, "rev": rev},
        steps=[
            Step(
                argv=["nix-env", "--profile", SYSTEM_PROFILE, "--set", toplevel],
                label=(
                    "point the system profile at this configuration, creating a "
                    "new generation"
                ),
                timeout=PROFILE_TIMEOUT,
            ),
            _switch_step(toplevel),
        ],
        summary=f"activate the NixOS configuration{provenance}",
        partial_detail=_R3_PARTIAL_DETAIL,
        cancel_detail=_R3_CANCEL_DETAIL,
    )


def _rollback_plan(args: RollbackArgs, *, runner: Runner, files: Files) -> Plan:
    listing = list_generations(runner, timeout=GENERATION_TIMEOUT)
    if not listing.ok:
        # Refuse rather than guess: a rollback names a generation, and without
        # the list there is no way to tell whether that generation exists, is the
        # one already running, or was collected.
        raise ActionRefused(
            "refusing a rollback while the generation list is unavailable "
            f"({listing.available.reason})"
        )
    target = next((g for g in listing.generations if g.number == args.generation), None)
    if target is None:
        available = ", ".join(str(g.number) for g in listing.generations[:20])
        raise ActionRefused(
            f"there is no generation {args.generation} on this system "
            f"(it has {available or 'none'})"
        )
    if target.current:
        raise ActionRefused(
            f"generation {args.generation} is the one already running, so "
            "rolling back to it would change nothing"
        )
    resolved = files.resolve(generation_link(args.generation))
    if not resolved:
        raise ActionRefused(
            f"generation {args.generation} is listed but "
            f"{generation_link(args.generation)} does not resolve, so there is "
            "no configuration to activate"
        )
    # The SAME validation an activate gets. A generation link is normally
    # trustworthy, but "it came from the profile directory" is not a reason to
    # skip the check that the thing is bootable.
    toplevel = validate_toplevel(resolved, runner=runner, files=files)
    described = " ".join(part for part in (target.date, target.nixos_version) if part)
    return Plan(
        kind=ActionKind.ROLLBACK,
        risk=RiskClass.R3,
        args={
            "generation": args.generation,
            "toplevel": toplevel,
        },
        steps=[
            Step(
                argv=[
                    "nix-env",
                    "--profile",
                    SYSTEM_PROFILE,
                    "--switch-generation",
                    # A generation number is digits, so it cannot be read as an
                    # option.
                    str(args.generation),
                ],
                label=f"point the system profile back at generation {args.generation}",
                timeout=PROFILE_TIMEOUT,
            ),
            _switch_step(toplevel),
        ],
        summary=(
            f"roll the system back to generation {args.generation}"
            + (f" ({described})" if described else "")
        ),
        partial_detail=_R3_PARTIAL_DETAIL,
        cancel_detail=_R3_CANCEL_DETAIL,
    )


def build_plan(
    kind: ActionKind,
    raw_args: dict[str, object],
    *,
    runner: Runner,
    files: Files = DEFAULT_FILES,
    now: datetime | None = None,
) -> Plan:
    """Validate an action and produce the exact commands that would run.

    Raises ``ActionRefused`` for anything this helper will not do. Nothing
    beyond this function may add, reorder or reinterpret an argv element.
    """
    risk = RISK_OF[kind]
    args = parse_args(kind, raw_args)

    if kind in UNIT_KINDS:
        assert isinstance(args, UnitArgs)
        unit = normalise_unit(args.unit)
        verb = _SYSTEMCTL_VERB[kind]
        return Plan(
            kind=kind,
            risk=risk,
            args={"unit": unit},
            steps=[
                Step(
                    # `--` before the positional so a name that slipped the
                    # charset check in some future edit still cannot be read as
                    # an option.
                    argv=["systemctl", verb, "--", unit],
                    label=f"{verb} {unit}",
                    timeout=120.0,
                )
            ],
            summary=f"{verb} the {unit} unit",
        )

    if kind is ActionKind.ACTIVATE:
        assert isinstance(args, ActivateArgs)
        return _activate_plan(args, runner=runner, files=files)

    if kind is ActionKind.ROLLBACK:
        assert isinstance(args, RollbackArgs)
        return _rollback_plan(args, runner=runner, files=files)

    if kind is ActionKind.GC_STORE:
        return Plan(
            kind=kind,
            risk=risk,
            args={},
            steps=[
                Step(
                    argv=nix_cli("store", "gc"),
                    label="delete every store path that is already unreachable",
                    # A full store collection is slow; it walks every path and
                    # holds the GC lock while it does.
                    timeout=3600.0,
                )
            ],
            summary="delete store paths that are already unreachable",
        )

    assert isinstance(args, GcOlderThanArgs)
    listing = list_generations(runner, timeout=GENERATION_TIMEOUT)
    if not listing.ok:
        # Refuse rather than proceed: without the generation list the floor
        # cannot be enforced, and the flag alone does not provide it.
        raise ActionRefused(
            "refusing an age-based collection while the generation list is "
            f"unavailable ({listing.available.reason}) - the two-generation "
            "floor cannot be checked, and --delete-older-than does not provide it"
        )
    moment = now or datetime.now()
    removed = generations_older_than(listing.generations, args.days, now=moment)
    if not removed:
        # Nothing qualifies. Refusing beats emitting a command that would do
        # nothing: an approval should never be asked for an empty act, and an
        # operator who approved one would learn nothing from the result.
        raise ActionRefused(
            f"no generation is both older than {args.days} days and outside the "
            f"{PROTECTED_GENERATIONS} most recent, so there is nothing to delete"
        )
    # The floor is in the ARGV, not just in the preview.
    #
    # This deliberately does NOT use `nix-collect-garbage --delete-older-than`.
    # That flag is purely age-based and keeps only the CURRENT generation, so on
    # a box whose previous generation is older than the cutoff it deletes the
    # exact rollback target the floor exists to protect - while the preview,
    # computed here, said it would be kept. Naming the generations makes the
    # command and the preview the same statement, and makes the floor a property
    # of what runs rather than of what was displayed.
    #
    # Generation numbers are digits, so they cannot be read as options.
    numbers = [str(generation.number) for generation in removed]
    return Plan(
        kind=kind,
        risk=risk,
        args={"days": args.days},
        steps=[
            Step(
                argv=[
                    "nix-env",
                    "--profile",
                    SYSTEM_PROFILE,
                    "--delete-generations",
                    *numbers,
                ],
                label=f"delete generation(s) {', '.join(numbers)}",
                timeout=3600.0,
            )
        ],
        summary=(
            f"delete {len(removed)} system generation(s) older than {args.days} "
            f"days ({', '.join(numbers)}), keeping "
            f"{len(listing.generations) - len(removed)}"
        ),
        generations_removed=[g.number for g in removed],
    )
