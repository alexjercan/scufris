"""Reading git, and addressing the build at an identified revision.

Nothing here edits, commits, branches or writes to the configuration repository:
it reads git and it builds. The build addresses the repository as
``git+file://<repo>?ref=<ref>&rev=<rev>``, so nix takes the tree from the COMMIT
rather than from the working tree - uncommitted files are structurally excluded
and this code cannot dirty the repository even by accident.
"""

from __future__ import annotations

import json
import re
import socket
from pathlib import Path

from scufris_host import Runner, nix_cli, run_command

from .models import ConfigChangeRefused, Resolved

# git reads are local and fast; the attribute probe is a flake evaluation.
GIT_TIMEOUT = 30.0
EVAL_TIMEOUT = 300.0

# A ref an operator or an agent might name: a branch, a tag or an object id. No
# leading dash (it would be read as an option), no whitespace, no `..` (which is
# a RANGE to git, not a commit), bounded length.
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@+-]{0,199}$")

# A ref that is already an object id rather than a name.
_REV_LIKE = re.compile(r"^[0-9a-fA-F]{7,64}$")

# What `--print-out-paths` prints, and the only thing this module will hand on as
# a toplevel. The helper validates it again, structurally, before it will name it
# in a privileged command.
_STORE_PATH = re.compile(r"^/nix/store/[0-9a-df-np-sv-z]{32}-[A-Za-z0-9._+=?-]{1,207}$")


def default_attr() -> str:
    """The nixosConfiguration this machine is, by name.

    The operator's flake names its hosts by directory, and this host's directory
    is its hostname - so the hostname is the right default and a wrong one is
    caught by the attribute probe with the real list.
    """
    return socket.gethostname()


def _git(runner: Runner, repo: Path, *args: str) -> tuple[bool, str]:
    result = runner(["git", "-C", str(repo), *args], timeout=GIT_TIMEOUT)
    return result.ok, result.stdout.strip()


def _validate_ref(ref: str) -> str:
    candidate = ref.strip()
    if not candidate:
        raise ConfigChangeRefused("no ref given: name a branch, tag or commit")
    if ".." in candidate:
        raise ConfigChangeRefused(
            f"refusing {candidate!r}: `..` is a git RANGE, not a commit"
        )
    if not _REF.match(candidate):
        raise ConfigChangeRefused(
            f"refusing {candidate!r}: a ref here is a branch, tag or object id "
            "in [A-Za-z0-9._/@+-], and may not start with '-'"
        )
    return candidate


def resolve(
    repo: Path,
    ref: str,
    *,
    runner: Runner = run_command,
    allowed: Path | None = None,
) -> tuple[Path, Resolved]:
    """Resolve ``ref`` in ``repo`` to a commit, and describe it honestly.

    Returns the MAIN repository path to build from alongside the resolution.
    ``repo`` may be a linked worktree (which is where an agent will have been
    working); commits are shared through the common object store, so the build
    addresses the main repository and gets the same tree.

    ``allowed``, when given, is the ONE repository this host's configuration may
    come from - checked against the resolved MAIN repository, so any worktree of
    it passes and nothing else does. Without it, a caller could name a repository
    it wrote itself and have Scufris build and propose that as the system: not as
    bad as handing over a store path, since the preview still names the
    repository and the revision, but the same shape one step removed. Which
    revision to build is a caller's business; which repository is not.
    """
    wanted = _validate_ref(ref)
    root = repo.expanduser()
    if not root.is_dir():
        raise ConfigChangeRefused(f"{root} is not a directory on this host")
    ok, common = _git(
        runner, root, "rev-parse", "--path-format=absolute", "--git-common-dir"
    )
    if not ok or not common:
        raise ConfigChangeRefused(f"{root} is not a git repository")
    main = Path(common).parent
    if not (main / "flake.nix").is_file():
        raise ConfigChangeRefused(
            f"{main} has no flake.nix, so it is not a NixOS configuration flake"
        )
    if allowed is not None:
        expected = allowed.expanduser()
        if main != expected:
            raise ConfigChangeRefused(
                f"refusing {root}: this host's configuration comes from "
                f"{expected} (or one of its worktrees), and {main} is not it. "
                "Which revision to build is yours to choose; which repository is "
                "not"
            )
    if wanted == "HEAD":
        # Resolve it to the branch it IS. "ref: HEAD @ 3af39d5" in an audit
        # record and an approval prompt says nothing; the branch name is what the
        # operator recognises, and it is what makes the flake reference honest.
        # A detached HEAD keeps the object id instead.
        ok_branch, branch = _git(runner, root, "rev-parse", "--abbrev-ref", "HEAD")
        if ok_branch and branch and branch != "HEAD":
            wanted = _validate_ref(branch)
    ok, rev = _git(runner, root, "rev-parse", "--verify", f"{wanted}^{{commit}}")
    if not ok or not rev:
        raise ConfigChangeRefused(
            f"{wanted!r} does not name a commit in {root} - has the branch been "
            "created and committed?"
        )
    _ok, subject = _git(runner, root, "log", "-1", "--format=%s", rev)
    _ok, head_branch = _git(runner, main, "rev-parse", "--abbrev-ref", "HEAD")
    merged, _out = _git(runner, main, "merge-base", "--is-ancestor", rev, "HEAD")
    ok_status, status = _git(runner, root, "status", "--porcelain")
    # `XY PATH`, and the leading space of an unstaged ` M path` is already gone -
    # _git strips - so split on the first space rather than slicing a column.
    uncommitted = (
        [
            line.strip().split(" ", 1)[1].strip()
            for line in status.splitlines()
            if line.strip() and " " in line.strip()
        ]
        if ok_status
        else []
    )
    return main, Resolved(
        repo=str(main),
        ref=wanted,
        rev=rev,
        subject=subject,
        merged=merged,
        head_branch=head_branch,
        uncommitted=uncommitted[:50],
    )


def flake_url(repo: Path, resolved: Resolved) -> str:
    """The flake reference that pins the build to one commit.

    ``?rev=`` alone is not enough when the commit is only on a branch, so the
    branch travels with it. A bare object id gets ``allRefs=1`` instead, which is
    what lets nix find a commit that is not the tip of anything.
    """
    if _REV_LIKE.match(resolved.ref):
        # A bare object id is not the tip of anything, so nix has to be told to
        # look past the default ref to find it.
        return f"git+file://{repo}?allRefs=1&rev={resolved.rev}"
    return f"git+file://{repo}?ref={resolved.ref}&rev={resolved.rev}"


def _attr_path(attr: str) -> str:
    return f"nixosConfigurations.{attr}.config.system.build.toplevel"


def build_argv(url: str, attr: str) -> list[str]:
    """The build, as the operator would type it.

    ``--no-link`` keeps a ``result`` symlink out of every directory (there is no
    working tree involved anyway), ``--print-out-paths`` is how the toplevel comes
    back, and the two lock-file flags make a stale ``flake.lock`` a FAILURE rather
    than something this build quietly updates: the lock belongs to the repository
    and updating it is a project change with its own review.
    """
    return nix_cli(
        "build",
        f"{url}#{_attr_path(attr)}",
        "--no-link",
        "--print-out-paths",
        "--no-update-lock-file",
        "--no-write-lock-file",
    )


def check_attr(url: str, attr: str, *, runner: Runner = run_command) -> None:
    """Refuse an unknown host attribute by NAME, listing the ones that exist."""
    result = runner(
        nix_cli(
            "eval",
            "--json",
            f"{url}#nixosConfigurations",
            "--apply",
            "builtins.attrNames",
        ),
        timeout=EVAL_TIMEOUT,
    )
    if not result.ok:
        raise ConfigChangeRefused(
            "this flake's nixosConfigurations could not be evaluated: "
            f"{result.reason()}"
        )
    try:
        parsed = json.loads(result.stdout or "[]")
    except ValueError as exc:
        raise ConfigChangeRefused(
            f"nix printed a host list this build cannot read: {exc}"
        ) from exc
    names = [str(name) for name in parsed] if isinstance(parsed, list) else []
    if attr not in names:
        raise ConfigChangeRefused(
            f"this flake has no nixosConfigurations.{attr} "
            f"(it has: {', '.join(n for n in names if n) or 'none'})"
        )


def toplevel_from(stdout: str) -> str:
    """The store path a successful build printed, or empty when it printed none."""
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if _STORE_PATH.match(candidate):
            return candidate
    return ""
