"""Every value that reaches an argv, held to a charset before it gets there.

This module owns the property the package exists to keep: **an argument may not
become a flag.** ``shell=False`` with an explicit argv answers a different
question - measured, a unit named ``-Hsomeone@host`` made systemctl open an
outbound SSH connection. Every value is charset-validated here, a leading ``-``
is refused explicitly, and the plan builders pass positionals after ``--``.

It also owns the R3 half of that property: the store path an activation names is
validated structurally before the helper will name it in a command, even though
the only code path that reaches it built that path itself.
"""

from __future__ import annotations

import re

from scufris_host import Runner, nix_cli

from ..files import Files
from .taxonomy import ActionRefused

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
# within it is a game of catch-up.
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

# A store path, and nothing but a store path: the 32-character nix base-32 hash
# (which has no e, o, t or u), a name, and NO further slash. A subpath like
# `<toplevel>/bin/switch-to-configuration` is not a system and must never be
# activatable, and `..` cannot survive this.
_STORE_PATH = re.compile(r"^/nix/store/[0-9a-df-np-sv-z]{32}-[A-Za-z0-9._+=?-]{1,207}$")

# A git revision, as recorded provenance. Never interpolated into a command - the
# app resolved it and built from it already - but it lands in the audit, so it is
# held to a charset like everything else that gets written there.
_REVISION = re.compile(r"^[0-9a-f]{7,64}$")

PATH_INFO_TIMEOUT = 60.0


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
    operator's reading of the diff are the real controls. What it does is make sure
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


def validate_provenance(repo: str, rev: str) -> tuple[str, str]:
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
