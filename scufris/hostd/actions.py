"""The action taxonomy: the verb set IS the risk classification.

Decided in ``tasks/20260729-125020/DECISION.md``. Five risk classes exist; three
of them have verbs here:

- **R1 service control** - ``unit_start``, ``unit_stop``, ``unit_restart``,
  ``unit_reload``. Reversible by restoring the recorded prior unit state.
- **R2 disposable cleanup** - ``gc_older_than``, ``gc_store``. One-way.
- **R3 declarative config change** - ``activate``, ``rollback``. Reversible by
  activating a recorded generation.

R0 needs no privilege and lives in ``scufris.host``. **R4 has no verb, and that
absence IS the enforcement** - partitioning, user and key material, the firewall,
and anything targeting scufris itself have no code path here rather than a deny
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

R3 adds a third, and it is the one the whole epic turns on: **the store path that
gets activated is not a caller's to choose.** ``activate`` takes a toplevel, but
the only code path that reaches it builds that path itself from a git revision it
resolved (``scufris/hostconfig.py``), the propose surfaces refuse the verb
outright, and this module still validates the path structurally before it will
name it in a command. See ``tasks/20260729-125035/DECISION.md`` section 2.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from ..host.run import Runner, nix_cli
from ..host.storage import Generation, list_generations
from .files import DEFAULT_FILES, Files

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

# --- R3 ---------------------------------------------------------------------

# A store path, and nothing but a store path: the 32-character nix base-32 hash
# (which has no e, o, t or u), a name, and NO further slash. A subpath like
# `<toplevel>/bin/switch-to-configuration` is not a system and must never be
# activatable, and `..` cannot survive this.
_STORE_PATH = re.compile(r"^/nix/store/[0-9a-df-np-sv-z]{32}-[A-Za-z0-9._+=?-]{1,207}$")

# A git revision, as recorded provenance. Never interpolated into a command - the
# app resolved it and built from it already - but it lands in the audit, so it is
# held to a charset like everything else that gets written there.
_REVISION = re.compile(r"^[0-9a-f]{7,64}$")

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
PATH_INFO_TIMEOUT = 60.0


class RiskClass(StrEnum):
    """Which class of the spike's taxonomy an action belongs to."""

    R1 = "r1"  # service control: reversible by restoring recorded state
    R2 = "r2"  # disposable cleanup: ONE-WAY
    R3 = "r3"  # declarative config change: reversible to a recorded generation


class ActionKind(StrEnum):
    """The complete set of verbs this helper implements. Nothing else exists."""

    UNIT_START = "unit_start"
    UNIT_STOP = "unit_stop"
    UNIT_RESTART = "unit_restart"
    UNIT_RELOAD = "unit_reload"
    GC_OLDER_THAN = "gc_older_than"
    GC_STORE = "gc_store"
    ACTIVATE = "activate"
    ROLLBACK = "rollback"


RISK_OF: dict[ActionKind, RiskClass] = {
    ActionKind.UNIT_START: RiskClass.R1,
    ActionKind.UNIT_STOP: RiskClass.R1,
    ActionKind.UNIT_RESTART: RiskClass.R1,
    ActionKind.UNIT_RELOAD: RiskClass.R1,
    ActionKind.GC_OLDER_THAN: RiskClass.R2,
    ActionKind.GC_STORE: RiskClass.R2,
    ActionKind.ACTIVATE: RiskClass.R3,
    ActionKind.ROLLBACK: RiskClass.R3,
}

UNIT_KINDS: frozenset[ActionKind] = frozenset(
    {
        ActionKind.UNIT_START,
        ActionKind.UNIT_STOP,
        ActionKind.UNIT_RESTART,
        ActionKind.UNIT_RELOAD,
    }
)

R3_KINDS: frozenset[ActionKind] = frozenset(
    {
        ActionKind.ACTIVATE,
        ActionKind.ROLLBACK,
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


class ActivateArgs(BaseModel):
    """Switch the system to an already-built configuration.

    ``toplevel`` is a store path the APP built from ``rev`` in ``repo``; the two
    provenance fields are recorded so the audit answers "which revision is this
    machine running" without trusting a description of it. They grant nothing and
    are never interpolated into a command.
    """

    toplevel: str
    repo: str = ""
    rev: str = ""


class RollbackArgs(BaseModel):
    """Return the system to a generation that already exists.

    A NUMBER, never a path: the helper resolves which store path that generation
    is, so "roll back" cannot be spelled as "activate this other thing".
    """

    generation: int = Field(ge=1)


ActionArgs = UnitArgs | GcOlderThanArgs | GcStoreArgs | ActivateArgs | RollbackArgs

_ARGS_MODEL: dict[ActionKind, type[BaseModel]] = {
    ActionKind.UNIT_START: UnitArgs,
    ActionKind.UNIT_STOP: UnitArgs,
    ActionKind.UNIT_RESTART: UnitArgs,
    ActionKind.UNIT_RELOAD: UnitArgs,
    ActionKind.GC_OLDER_THAN: GcOlderThanArgs,
    ActionKind.GC_STORE: GcStoreArgs,
    ActionKind.ACTIVATE: ActivateArgs,
    ActionKind.ROLLBACK: RollbackArgs,
}


class Step(BaseModel):
    """One command in a plan, with the wall clock it is allowed and a label.

    Steps exist because activation is not one command: the system profile is
    pointed at the built configuration, and THEN that configuration is switched
    to. Modelling that as a sequence is what lets the record say which half
    happened when the second one fails - see ``Plan.partial_detail``.
    """

    argv: list[str]
    # What this step does, in the operator's language. Rendered next to the
    # command in the preview and carried into the audit.
    label: str = ""
    timeout: float = 60.0


class Plan(BaseModel):
    """A validated action: what will run, and what the operator is agreeing to.

    Every ``Step.argv`` is built HERE, from the verb and the validated
    arguments, and carried on the plan so the preview, the audit record and the
    execution all name the same commands - the operator approves an argv, not a
    description of one.
    """

    kind: ActionKind
    risk: RiskClass
    args: dict[str, object] = Field(default_factory=dict)
    steps: list[Step]
    # A one-line statement of what this does, in the operator's language.
    summary: str
    # Set for R2: the generations this action would remove, resolved before the
    # command runs so the floor is enforced by us and not by a flag.
    generations_removed: list[int] = Field(default_factory=list)
    # What it means when a step after the first one fails, when that is not
    # simply "nothing happened". Empty for a single-step plan.
    partial_detail: str = ""
    # What a cancellation actually achieves for this class, when it is not "the
    # process group was signalled". R3 sets it because the switch runs in a
    # transient systemd unit that outlives this helper by design.
    cancel_detail: str = ""

    @property
    def argvs(self) -> list[list[str]]:
        """Every command, in order. Convenience for rendering."""
        return [step.argv for step in self.steps]


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


def validate_toplevel(candidate: str, *, runner: Runner, files: Files) -> str:
    """Refuse anything that is not a built NixOS system in this store.

    Four questions, each with its own answer, because "that path is not
    activatable" is not one failure:

    1. is it shaped like a store path ROOT (a subpath is not a system),
    2. is it a path this store actually knows (registered and valid),
    3. does it carry a ``nixos-version`` - which is `nixos-rebuild`'s OWN guard
       before it points the profile at anything, and it exists because a path
       that lacks it can leave the machine unable to boot,
    4. does it carry the ``switch-to-configuration`` we are about to run.

    None of this makes an activation SAFE - a NixOS system can contain any
    activation script its author wrote, which is why the reviewed commit and the
    operator's reading of the diff are the real controls
    (``tasks/20260729-125035/DECISION.md`` section 3). What it does is make sure
    the thing being activated is a system at all.
    """
    path = candidate.strip()
    if not path:
        raise ActionRefused("no configuration path given")
    if not _STORE_PATH.match(path):
        raise ActionRefused(
            f"refusing {path!r}: an activation names the ROOT of a store path "
            "(/nix/store/<hash>-<name>) and nothing else - not a subpath, not a "
            "symlink outside the store, not a relative path"
        )
    known = runner(nix_cli("path-info", "--", path), timeout=PATH_INFO_TIMEOUT)
    if not known.ok:
        raise ActionRefused(
            f"refusing {path}: this store does not have it as a valid path "
            f"({known.reason()})"
        )
    if not files.is_file(f"{path}/nixos-version"):
        raise ActionRefused(
            f"refusing {path}: it has no nixos-version, so it is not a built "
            "NixOS system. This is nixos-rebuild's own precondition before it "
            "points the system profile at a path, and skipping it is how a "
            "machine ends up unable to boot"
        )
    if not files.is_executable(f"{path}/bin/switch-to-configuration"):
        raise ActionRefused(
            f"refusing {path}: it has no bin/switch-to-configuration to run"
        )
    return path


def _validate_provenance(repo: str, rev: str) -> tuple[str, str]:
    """Hold the recorded provenance to a charset before it enters the audit."""
    clean_repo = repo.strip()
    if clean_repo:
        if not clean_repo.startswith("/") or len(clean_repo) > 4096:
            raise ActionRefused(
                f"the recorded repository must be an absolute path, got {clean_repo!r}"
            )
        if any(char in clean_repo for char in "\n\r\x00"):
            raise ActionRefused("the recorded repository contains control characters")
    clean_rev = rev.strip().lower()
    if clean_rev and not _REVISION.match(clean_rev):
        raise ActionRefused(
            f"the recorded revision must be a git object id, got {rev!r}"
        )
    return clean_repo, clean_rev


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
    repo, rev = _validate_provenance(args.repo, args.rev)
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
    # of what runs rather than of what was displayed (review round 1, R1.4).
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
