"""The action taxonomy: the verb set IS the risk classification.

Decided in ``tasks/20260729-125020/DECISION.md``. Five risk classes exist; two
of them have verbs here:

- **R1 service control** - ``unit_start``, ``unit_stop``, ``unit_restart``,
  ``unit_reload``. Reversible by restoring the recorded prior unit state.
- **R2 disposable cleanup** - ``gc_older_than``, ``gc_store``. One-way.

R0 needs no privilege and lives in ``scufris.host``. R3 (the NixOS
configuration change) is 20260729-125035. **R4 has no verb, and that absence IS
the enforcement** - partitioning, user and key material, the firewall, and
anything targeting scufris itself have no code path here rather than a deny
check that could have a bug.

Two properties this module is responsible for, both of which have already been
paid for once in this repo:

1. **The helper builds every argv.** A caller names a verb and typed arguments;
   it never supplies a command. There is no shell verb at any privilege under
   any approval.
2. **An argument may not become a flag.** ``shell=False`` with an explicit argv
   answers a different question - measured in 20260729-125024, a unit named
   ``-Hsomeone@host`` made systemctl open an outbound SSH connection. Every
   value is charset-validated, a leading ``-`` is refused explicitly, and
   positionals are passed after ``--``.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from ..host.run import Runner
from ..host.storage import Generation, list_generations

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

# Unit names systemd accepts, restricted to what an operator actually types. No
# leading dash (the option-injection guard), no slash (a path is not a unit
# name), no whitespace, bounded length.
_UNIT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_.@-]{0,127}$")

# Unit suffixes this helper will act on. A name with no suffix is normalised to
# `.service`, which is what an operator means by "restart nginx". Anything else
# must be spelled out, so a typo cannot silently resolve to a different unit
# type.
#
# `.target`, `.slice` and `.scope` are deliberately ABSENT. They are not units
# an operator means when they say "restart that service", and they are how the
# deny-list gets walked around: `unit_start emergency.target` drops the box to
# single-user and kills sshd - the exact outcome the `sshd` entry exists to
# prevent, through a name no stem list would catch - while `unit_stop
# user.slice` or `user@1000.service` kills the scufris USER service itself.
# Refusing the whole unit TYPE is a boundary; enumerating the dangerous names
# within it is a game of catch-up (review round 1, R1.5).
_UNIT_SUFFIXES = (
    ".service",
    ".socket",
    ".timer",
    ".path",
    ".mount",
)

# Unit types that exist but that this helper will not act on, kept separate from
# "unknown suffix" so the refusal can say WHY rather than "unrecognised".
_REFUSED_UNIT_SUFFIXES = (
    ".target",
    ".slice",
    ".scope",
)

# R1's deny-list: units whose loss takes out the operator's remote access, the
# desktop session, or the approval path itself. Compared on the STEM (the name
# without its suffix), case-insensitively, because `sshd` and `sshd.service` are
# the same daemon.
#
# This list is the SECOND line. The first is the suffix restriction above: the
# most dangerous names are targets and slices, and those have no code path at
# all now.
DENIED_UNIT_STEMS: frozenset[str] = frozenset(
    {
        "sshd",
        "ssh",
        "dbus",
        "dbus-broker",
        "systemd-logind",
        "networkmanager",
        "systemd-networkd",
        "polkit",
        # The operator's own session and its manager - killing these logs them
        # out and takes the scufris USER service with them.
        "user",
        "user-runtime-dir",
        "systemd-user-sessions",
        "display-manager",
        "getty",
        "serial-getty",
        # pid 1's own plumbing: the journal IS the audit's neighbour, and
        # nix-daemon is what every remaining verb needs to do anything.
        "init",
        "systemd-journald",
        "systemd-udevd",
        "dbus-daemon",
        "nix-daemon",
    }
)

# A templated instance (`user@1000.service`) shares its danger with the
# template, so the deny check compares the TEMPLATE part - everything before the
# `@` - as well as the whole stem.
_INSTANCE_SEPARATOR = "@"

# Refused regardless of the deny-list above: the helper must not be able to act
# on scufris or on itself. A verb that can restart the approval path is a verb
# that can end an approval mid-flight, and one that can restart the helper can
# drop the audit record of what it was doing.
_SELF_MARKER = "scufris"


class RiskClass(StrEnum):
    """Which class of the spike's taxonomy an action belongs to."""

    R1 = "r1"  # service control: reversible by restoring recorded state
    R2 = "r2"  # disposable cleanup: ONE-WAY


class ActionKind(StrEnum):
    """The complete set of verbs this helper implements. Nothing else exists."""

    UNIT_START = "unit_start"
    UNIT_STOP = "unit_stop"
    UNIT_RESTART = "unit_restart"
    UNIT_RELOAD = "unit_reload"
    GC_OLDER_THAN = "gc_older_than"
    GC_STORE = "gc_store"


RISK_OF: dict[ActionKind, RiskClass] = {
    ActionKind.UNIT_START: RiskClass.R1,
    ActionKind.UNIT_STOP: RiskClass.R1,
    ActionKind.UNIT_RESTART: RiskClass.R1,
    ActionKind.UNIT_RELOAD: RiskClass.R1,
    ActionKind.GC_OLDER_THAN: RiskClass.R2,
    ActionKind.GC_STORE: RiskClass.R2,
}

UNIT_KINDS: frozenset[ActionKind] = frozenset(
    {
        ActionKind.UNIT_START,
        ActionKind.UNIT_STOP,
        ActionKind.UNIT_RESTART,
        ActionKind.UNIT_RELOAD,
    }
)

_SYSTEMCTL_VERB: dict[ActionKind, str] = {
    ActionKind.UNIT_START: "start",
    ActionKind.UNIT_STOP: "stop",
    ActionKind.UNIT_RESTART: "restart",
    ActionKind.UNIT_RELOAD: "reload",
}


class ActionRefused(Exception):
    """An action this helper will not build an argv for.

    Raised by validation, never by execution: by the time an action has a
    ``Plan`` it is a command the helper is willing to run once approved.
    """


class UnitArgs(BaseModel):
    """The single argument every R1 verb takes."""

    unit: str


class GcOlderThanArgs(BaseModel):
    """Trim system generations (and the store paths they held) by age."""

    # Bounded on both ends: 0 would mean "everything including today", and a
    # value past ten years is a typo rather than an intent.
    days: int = Field(ge=1, le=3650)


class GcStoreArgs(BaseModel):
    """Delete store paths that are already dead. Touches no generation."""


ActionArgs = UnitArgs | GcOlderThanArgs | GcStoreArgs

_ARGS_MODEL: dict[ActionKind, type[BaseModel]] = {
    ActionKind.UNIT_START: UnitArgs,
    ActionKind.UNIT_STOP: UnitArgs,
    ActionKind.UNIT_RESTART: UnitArgs,
    ActionKind.UNIT_RELOAD: UnitArgs,
    ActionKind.GC_OLDER_THAN: GcOlderThanArgs,
    ActionKind.GC_STORE: GcStoreArgs,
}


class Plan(BaseModel):
    """A validated action: what will run, and what the operator is agreeing to.

    ``argv`` is built HERE, from the verb and the validated arguments. It is
    carried on the plan so the preview, the audit record and the execution all
    name the same command - the operator approves an argv, not a description of
    one.
    """

    kind: ActionKind
    risk: RiskClass
    args: dict[str, object] = Field(default_factory=dict)
    argv: list[str]
    # A one-line statement of what this does, in the operator's language.
    summary: str
    # Set for R2: the generations this action would remove, resolved before the
    # command runs so the floor is enforced by us and not by a flag.
    generations_removed: list[int] = Field(default_factory=list)
    # How long execution may take. GC walks the whole store.
    timeout: float = 60.0


def parse_args(kind: ActionKind, raw: dict[str, object]) -> BaseModel:
    """Validate ``raw`` against the argument model for ``kind``.

    A verb that is not in the table cannot get here - ``ActionKind`` is a
    closed enum and an unknown string fails to parse at the protocol boundary.
    """
    model = _ARGS_MODEL[kind]
    try:
        return model.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - a pydantic error is a refusal
        raise ActionRefused(f"invalid arguments for {kind}: {exc}") from exc


def normalise_unit(name: str) -> str:
    """Validate a unit name and give it an explicit suffix.

    The three refusals here are the option-injection guard, and each is a
    separate question: is it in the charset, does it start with a dash, and is
    it a unit type we act on.
    """
    candidate = name.strip()
    if not candidate:
        raise ActionRefused("no unit name given")
    if candidate.startswith("-"):
        # Belt and braces with the charset below: a value starting with `-` is
        # parsed as a FLAG by systemctl even with shell=False and even after a
        # `--` separator would have protected it, so it is refused by name.
        raise ActionRefused(
            f"refusing a unit name that starts with '-': {candidate!r} "
            "would be read as an option, not a unit"
        )
    if not _UNIT_NAME.match(candidate):
        raise ActionRefused(
            f"refusing a unit name outside the allowed charset: {candidate!r}"
        )
    if candidate.endswith(_REFUSED_UNIT_SUFFIXES):
        raise ActionRefused(
            f"refusing to act on {candidate}: this helper controls services, "
            "sockets, timers, paths and mounts - not targets, slices or scopes. "
            "Those change what the whole machine is running (emergency.target "
            "drops it to single-user; user.slice ends the operator's session), "
            "which is not something a service verb should reach"
        )
    if not candidate.endswith(_UNIT_SUFFIXES):
        if "." in candidate:
            raise ActionRefused(
                f"refusing an unknown unit type: {candidate!r} "
                f"(expected one of {', '.join(_UNIT_SUFFIXES)})"
            )
        candidate = f"{candidate}.service"
    stem = candidate.rsplit(".", 1)[0].lower()
    # A templated instance is as dangerous as its template, so both forms are
    # checked: `user@1000.service` must be refused by the `user` entry.
    template = stem.split(_INSTANCE_SEPARATOR, 1)[0]
    if _SELF_MARKER in stem:
        raise ActionRefused(
            f"refusing to act on {candidate}: the helper may not control "
            "scufris or itself - that would let an action end the approval "
            "path or the record of what it was doing"
        )
    if stem in DENIED_UNIT_STEMS or template in DENIED_UNIT_STEMS:
        raise ActionRefused(
            f"refusing to act on {candidate}: it is on the deny-list, because "
            "losing it takes out remote access or the session this approval "
            "arrived through"
        )
    return candidate


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


def build_plan(
    kind: ActionKind,
    raw_args: dict[str, object],
    *,
    runner: Runner,
    now: datetime | None = None,
) -> Plan:
    """Validate an action and produce the exact command that would run.

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
            # `--` before the positional so a name that slipped the charset
            # check in some future edit still cannot be read as an option.
            argv=["systemctl", verb, "--", unit],
            summary=f"{verb} the {unit} unit",
            timeout=120.0,
        )

    if kind is ActionKind.GC_STORE:
        return Plan(
            kind=kind,
            risk=risk,
            args={},
            argv=["nix", "store", "gc"],
            summary="delete store paths that are already unreachable",
            # A full store collection is slow; it walks every path and holds the
            # GC lock while it does.
            timeout=3600.0,
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
    # of what runs rather than of what was displayed (review round 1, R1.4).
    #
    # Generation numbers are digits, so they cannot be read as options.
    numbers = [str(generation.number) for generation in removed]
    return Plan(
        kind=kind,
        risk=risk,
        args={"days": args.days},
        argv=[
            "nix-env",
            "--profile",
            SYSTEM_PROFILE,
            "--delete-generations",
            *numbers,
        ],
        summary=(
            f"delete {len(removed)} system generation(s) older than {args.days} "
            f"days ({', '.join(numbers)}), keeping "
            f"{len(listing.generations) - len(removed)}"
        ),
        generations_removed=[g.number for g in removed],
        timeout=3600.0,
    )
