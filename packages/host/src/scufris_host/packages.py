"""Package and generation queries: what is installed, and what changed.

Four questions, all answerable without privilege:

- what provides ``<binary>`` - resolved on PATH, then through symlinks to the
  store path, whose name IS the package identity on NixOS;
- what is in the profiles - the system profile (1155 binaries on this host) and
  the home-manager/user profile;
- how two generations differ - ``nix store diff-closures``, which is builtin to
  the nix on this host so no ``nvd`` dependency is added anywhere;
- how old the config repo's flake inputs are - read from ``flake.lock``.

The closure diff carries the trap the host spike measured and this module is
where it is defused: when the two closures are identical, ``nix store
diff-closures`` prints NOTHING and exits 0, so "no change" and "the command
broke" are byte-identical in its output. This module branches on the exit status
and returns ``identical=True``, never an empty diff that a renderer could show as
a blank panel.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .models import Availability, Bounded, Report, clamp
from .run import Runner

SYSTEM_PROFILE = Path("/run/current-system/sw")
SYSTEM_PROFILES_DIR = Path("/nix/var/nix/profiles")

DEFAULT_PACKAGE_LIMIT = 40
MAX_PACKAGE_LIMIT = 500
# A closure diff over a large generation gap can list hundreds of packages.
DEFAULT_DIFF_LIMIT = 60
MAX_DIFF_LIMIT = 400
CLOSURE_DIFF_TIMEOUT = 60.0

# /nix/store/<32-char hash>-<name>-<version>, and nothing below it. The `[^/]+`
# matters: this is matched against every ancestor of a resolved binary, and a
# permissive `.+` matches the FULL path too, yielding the package name
# "systemd-261/bin/systemctl" (caught by running the example against the real
# host). The version is optional and the name can itself contain dashes, so the
# version is only split off when the last segment starts with a digit.
_STORE_PATH = re.compile(r"^/nix/store/[a-z0-9]{32}-(?P<name>[^/]+)$")

# `nix store diff-closures` output lines, e.g.
#   linux: 6.18.37 -> 6.18.40
#   foo: (absent) -> 1.2
#   bar: 1.0 -> (absent), -2.3 MiB
# (nix writes the empty-set glyph where "(absent)" is written here; the repo is
# plain ASCII, and the parser does not depend on that character either way.)
_DIFF_LINE = re.compile(r"^\s*(?P<name>[^:]+):\s*(?P<detail>.+)$")


class BinaryProvider(Report):
    """Which package provides a binary."""

    binary: str
    path: str = ""
    store_path: str = ""
    package: str = ""
    version: str = ""
    # True when the binary resolves outside the nix store (a script in ~/bin).
    outside_store: bool = False


class ProfileReport(Report):
    """What a profile contains."""

    profile: str
    binary_count: int = 0
    store_path: str = ""
    packages: list[str] = Field(default_factory=list)


class ClosureChange(BaseModel):
    package: str
    detail: str


class ClosureDiff(Report, Bounded):
    """The difference between two system closures.

    ``identical`` distinguishes "the two closures are the same" from "the diff
    could not be produced" - the two cases nix itself renders identically.
    """

    before: str = ""
    after: str = ""
    identical: bool = False
    changes: list[ClosureChange] = Field(default_factory=list)


class FlakeInput(BaseModel):
    name: str
    revision: str = ""
    last_modified: datetime | None = None
    source: str = ""

    def age_days(self, *, now: datetime | None = None) -> int | None:
        if self.last_modified is None:
            return None
        moment = now or datetime.now(timezone.utc)
        return max(0, (moment - self.last_modified).days)


class FlakeStatus(Report):
    """How old the config repo's pinned inputs are.

    This reads ``flake.lock`` and reports each input's pin AGE. It deliberately
    does NOT claim an input is "behind": proving a newer commit exists needs a
    network fetch (``nix flake metadata --refresh``), which an inspection tool
    should not perform silently. The caveat says so rather than letting age be
    misread as staleness.
    """

    repository: str = ""
    inputs: list[FlakeInput] = Field(default_factory=list)

    def oldest(self) -> FlakeInput | None:
        dated = [i for i in self.inputs if i.last_modified is not None]
        return min(dated, key=lambda i: i.last_modified or datetime.max, default=None)


def _split_name_version(raw: str) -> tuple[str, str]:
    """Split a store path's ``<name>-<version>`` tail, tolerating dashes in the
    name. A trailing segment counts as a version only if it starts with a digit."""
    head, sep, tail = raw.rpartition("-")
    if sep and tail and tail[0].isdigit():
        return head, tail
    return raw, ""


def what_provides(binary: str) -> BinaryProvider:
    """Resolve a binary on PATH to the store package that provides it."""
    import shutil

    name = binary.strip()
    if not name or "/" in name:
        return BinaryProvider(
            binary=binary,
            available=Availability.unavailable(
                "pass a bare command name, not a path (e.g. 'systemctl')"
            ),
        )
    found = shutil.which(name)
    if found is None:
        return BinaryProvider(
            binary=name,
            available=Availability.unavailable(
                f"{name!r} is not on PATH, so nothing on this host provides it "
                "as a command"
            ),
        )
    report = BinaryProvider(binary=name, path=found)
    try:
        real = Path(found).resolve()
    except OSError as exc:
        report.available = Availability(
            ok=True,
            caveat=f"the symlink chain for {found} could not be followed: {exc}",
        )
        return report
    report.store_path = str(real)
    # Walk up to the store root: /nix/store/<hash>-<pkg>/bin/foo -> the <pkg> dir.
    for parent in [real, *real.parents]:
        match = _STORE_PATH.match(str(parent))
        if match is None:
            continue
        report.store_path = str(parent)
        report.package, report.version = _split_name_version(match.group("name"))
        return report
    report.outside_store = True
    report.available = Availability(
        ok=True,
        caveat=(
            f"{found} resolves to {real}, which is outside the nix store - it is "
            "not provided by a package"
        ),
    )
    return report


def profile_contents(
    *,
    profile: Path = SYSTEM_PROFILE,
    limit: int = DEFAULT_PACKAGE_LIMIT,
) -> ProfileReport:
    """What a profile provides, by listing the binaries it exposes.

    Listing binaries rather than derivations is deliberate: the system profile
    has no manifest to read, and "which commands do I have" is the question an
    operator actually asks.
    """
    capped = clamp(limit, default=DEFAULT_PACKAGE_LIMIT, maximum=MAX_PACKAGE_LIMIT)
    report = ProfileReport(profile=str(profile))
    binaries = profile / "bin"
    if not binaries.is_dir():
        report.available = Availability.unavailable(
            f"{binaries} does not exist, so this profile has no binaries to list"
        )
        return report
    try:
        names = sorted(entry.name for entry in binaries.iterdir())
    except OSError as exc:
        report.available = Availability.unavailable(
            f"{binaries} could not be listed: {exc}"
        )
        return report
    try:
        report.store_path = str(profile.resolve())
    except OSError:
        report.store_path = ""
    report.binary_count = len(names)
    report.packages = names[:capped]
    return report


def _generation_link(generation: int) -> Path:
    return SYSTEM_PROFILES_DIR / f"system-{generation}-link"


def closure_diff(
    runner: Runner,
    before: str | int,
    after: str | int,
    *,
    limit: int = DEFAULT_DIFF_LIMIT,
    timeout: float = CLOSURE_DIFF_TIMEOUT,
) -> ClosureDiff:
    """Diff two closures, given generation NUMBERS or store paths.

    Empty output with a zero exit means the closures are identical - the trap the
    spike measured live. It is reported as ``identical=True`` and never as an
    empty change list, so a renderer cannot show "no change" and "the diff broke"
    as the same blank.
    """
    capped = clamp(limit, default=DEFAULT_DIFF_LIMIT, maximum=MAX_DIFF_LIMIT)

    def resolve(value: str | int) -> str | None:
        """A generation number or a store path, or None if it is neither.

        `nix store diff-closures` takes INSTALLABLES, not just store paths, so an
        unvalidated string here would let a caller name a flake output
        ("nixpkgs#firefox") and make this read-only inspection fetch and BUILD
        it. The MCP tool's signature is `int`, but this method is public API and
        the guarantee has to live in the layer that claims it, not in a caller's
        discipline.
        """
        text = str(value)
        if isinstance(value, int) or text.isdigit():
            return str(_generation_link(int(text)))
        # Accept a store path, and the profile-link form this same function
        # produces for an int - otherwise passing back a value it just returned
        # would be refused.
        allowed = (f"{SYSTEM_PROFILES_DIR}/", "/nix/store/")
        if text.startswith(allowed) and Path(text).exists():
            return text
        return None

    before_path = resolve(before)
    after_path = resolve(after)
    report = ClosureDiff(
        before=before_path or str(before), after=after_path or str(after), limit=capped
    )
    if before_path is None or after_path is None:
        bad = before if before_path is None else after
        report.available = Availability.unavailable(
            f"{bad!r} is not a generation number or an existing /nix/store path; "
            "this tool diffs closures and will not evaluate a flake installable"
        )
        return report
    result = runner(
        ["nix", "store", "diff-closures", "--", before_path, after_path],
        timeout=timeout,
    )
    report.available = Availability.of(result)
    if not result.ok:
        return report
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        # THE trap: exit 0 with no output. Say it plainly.
        report.identical = True
        return report
    changes: list[ClosureChange] = []
    for line in lines:
        match = _DIFF_LINE.match(line)
        if match is None:
            continue
        changes.append(
            ClosureChange(
                package=match.group("name").strip(),
                detail=match.group("detail").strip(),
            )
        )
    report.total_seen = len(changes)
    report.changes = changes[:capped]
    report.truncated = len(changes) > capped
    return report


def flake_status(repository: Path) -> FlakeStatus:
    """Read ``flake.lock`` in ``repository`` and report each input's pin age."""
    repo = repository.expanduser()
    report = FlakeStatus(repository=str(repo))
    lock = repo / "flake.lock"
    if not lock.is_file():
        report.available = Availability.unavailable(
            f"no flake.lock at {lock} - set SCUFRIS_HOST_CONFIG_REPO to the "
            "directory holding this host's NixOS flake"
        )
        return report
    try:
        data = json.loads(lock.read_text())
    except (OSError, ValueError) as exc:
        report.available = Availability.unavailable(
            f"{lock} could not be read as a flake lock: {exc}"
        )
        return report
    nodes = data.get("nodes") if isinstance(data, dict) else None
    if not isinstance(nodes, dict):
        report.available = Availability.unavailable(
            f"{lock} has no 'nodes' table; this build cannot read that lock format"
        )
        return report
    root_name = data.get("root", "root")
    root = nodes.get(root_name, {})
    # Only the ROOT flake's own inputs. The transitive set (flake-parts_2,
    # nixpkgs-lib_3, ...) is dozens of nodes the operator never pinned.
    direct = root.get("inputs", {}) if isinstance(root, dict) else {}
    for name, node_key in sorted(direct.items()):
        key = node_key[0] if isinstance(node_key, list) and node_key else node_key
        node = nodes.get(key) if isinstance(key, str) else None
        locked = node.get("locked", {}) if isinstance(node, dict) else {}
        if not isinstance(locked, dict):
            continue
        stamp = locked.get("lastModified")
        moment: datetime | None = None
        if isinstance(stamp, (int, float)):
            try:
                moment = datetime.fromtimestamp(float(stamp), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                moment = None
        owner = str(locked.get("owner", ""))
        repo_name = str(locked.get("repo", ""))
        report.inputs.append(
            FlakeInput(
                name=name,
                revision=str(locked.get("rev", "")),
                last_modified=moment,
                source=f"{owner}/{repo_name}"
                if owner and repo_name
                else str(locked.get("url", locked.get("type", ""))),
            )
        )
    if not report.inputs:
        report.available = Availability.unavailable(
            f"{lock} declares no direct inputs for its root flake"
        )
        return report
    report.available = Availability(
        ok=True,
        caveat=(
            "these are PIN AGES read from flake.lock. Whether a newer commit "
            "exists upstream needs a network fetch, which this read-only "
            "inspection does not perform"
        ),
    )
    return report
