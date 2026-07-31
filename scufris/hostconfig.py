"""Turning a reviewed commit into a configuration the operator can approve.

This is the unprivileged half of the R3 flow, and the division of labour is the
whole design:

- The configuration repository is **a project**. An agent changes it the way an
  agent changes any project - a worktree, a commit, a review. Nothing in this
  module edits, commits, branches or writes to it. It reads git and it builds.
- The build runs as the OPERATOR, never as root. Nix evaluation reads files with
  the evaluating user's privileges, so a configuration evaluated as root could
  read a host key or a sops age key into a derivation output; as ``alex`` that
  read simply fails.
- The store path that gets activated is built HERE, from a revision resolved
  HERE. A caller never supplies one - ``/api/host/actions`` refuses
  ``kind=activate`` outright - because otherwise the model would choose what gets
  activated and the closure diff would faithfully describe whatever it chose.

The build addresses the repository as ``git+file://<repo>?ref=<ref>&rev=<rev>``,
which is not a detail: nix then takes the tree from the COMMIT rather than from
the working tree, so what gets built is an identified revision, uncommitted files
are structurally excluded (and said so, rather than silently ignored), and this
module cannot dirty the repository even by accident.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import socket
import time
from collections import OrderedDict
from enum import StrEnum
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from .eventbus import EventBus
from .host.run import Outcome, Runner, nix_cli, run_command
from .hostd.executor import Executor, run_action
from .supervisor import Supervisor

logger = logging.getLogger(__name__)

# Bounded like the action registry: config changes are short-lived records and
# the audit log is the durable half.
MAX_CHANGES = 100

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

# How much of a failed build's log is kept on the record. A nix build failure is
# chatty and the useful part is the end.
MAX_LOG_TAIL = 16000


class ConfigChangeRefused(Exception):
    """Something about the request or the repository makes a build impossible.

    Always carries a sentence an operator can act on - "that ref does not exist
    in that repository", not "git failed".
    """


class ChangeState(StrEnum):
    """Where a configuration change is in its life.

    ``FAILED`` and ``CANCELLED`` are terminal and carry no proposal: a change
    that did not build cannot be approved, because there is nothing to activate.
    """

    BUILDING = "building"
    PROPOSED = "proposed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Resolved(BaseModel):
    """What a ref means right now, and what the operator should know about it."""

    repo: str
    ref: str
    rev: str
    subject: str = ""
    # Whether the revision is already contained in what the operator's own
    # checkout has. A change built from a branch that is not merged yet is
    # normal and fine - it just means merging it back is still to do, which is a
    # project act and not something this flow does.
    merged: bool | None = None
    head_branch: str = ""
    # Files modified in the working tree the ref was resolved from. They are NOT
    # in the build, by construction, and saying so is the difference between an
    # honest preview and an agent wondering why its edit did nothing.
    uncommitted: list[str] = Field(default_factory=list)


class ConfigChange(BaseModel):
    """One configuration change as the API and the dashboard see it."""

    id: str
    resolved: Resolved
    attr: str
    state: ChangeState = ChangeState.BUILDING
    # The built system, once there is one.
    toplevel: str = ""
    # The host action proposal that carries the activation, once it exists. This
    # is the id every approval, denial and audit record uses.
    action_id: str = ""
    run_id: str = ""
    log_tail: str = ""
    error: str = ""
    created_at: float = 0.0
    # Which agent asked, when one did. Recorded for display; it grants nothing.
    agent: str = ""
    requested_by: str = ""

    @property
    def repo(self) -> str:
        return self.resolved.repo


# --- what a build publishes on the bus -----------------------------------
#
# Its own event type, like host applies: a build log is not model text and a
# built configuration is not an applied action.


class ConfigBuildOutput(BaseModel):
    type: Literal["output"] = "output"
    stream: str
    text: str


class ConfigBuildDone(BaseModel):
    type: Literal["done"] = "done"
    change: ConfigChange


class ConfigBuildError(BaseModel):
    type: Literal["error"] = "error"
    detail: str


ConfigBuildEvent = ConfigBuildOutput | ConfigBuildDone | ConfigBuildError

ConfigSupervisor = Supervisor[ConfigBuildEvent]
ConfigBuildBus = EventBus[ConfigBuildEvent]


def _build_error_event(detail: str) -> ConfigBuildEvent:
    return ConfigBuildError(detail=detail)


def _build_error_detail(event: ConfigBuildEvent) -> str | None:
    return event.detail if isinstance(event, ConfigBuildError) else None


def config_supervisor(
    *,
    max_concurrent: int = 1,
    max_history: int = 50,
    clock: Callable[[], float] = time.time,
) -> ConfigSupervisor:
    """A supervisor for configuration builds.

    Separate from the host-apply supervisor on purpose. A NixOS build can run for
    a long time and needs no privilege; sharing the single apply slot with it
    would mean a kernel rebuild blocks an unrelated service restart the operator
    approved.
    """
    return Supervisor(
        error_event=_build_error_event,
        error_detail=_build_error_detail,
        max_concurrent=max_concurrent,
        max_history=max_history,
        clock=clock,
    )


# --- git ------------------------------------------------------------------


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


# --- the store -------------------------------------------------------------


class UnknownChange(KeyError):
    """No such configuration change id."""


class ChangeInFlight(RuntimeError):
    """A build is already running against this repository.

    Refused rather than queued, and the difference matters to the person
    waiting: a queued NixOS build can sit for an hour behind another one with no
    sign of why. Two builds of the same repository also contend for the same
    evaluation and the same store, so serializing them buys nothing but
    confusion.
    """


class ConfigChangeStore:
    """The app's bounded registry of configuration changes."""

    def __init__(
        self,
        *,
        max_changes: int = MAX_CHANGES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._changes: "OrderedDict[str, ConfigChange]" = OrderedDict()
        self._max = max_changes
        self._now = clock

    def put(self, change: ConfigChange) -> ConfigChange:
        if not change.created_at:
            change.created_at = self._now()
        self._changes[change.id] = change
        self._reap()
        return change

    def get(self, change_id: str) -> ConfigChange:
        try:
            return self._changes[change_id]
        except KeyError as exc:
            raise UnknownChange(change_id) from exc

    def list(self) -> list[ConfigChange]:
        return list(reversed(self._changes.values()))

    def building_for(self, repo: str) -> ConfigChange | None:
        return next(
            (
                change
                for change in self._changes.values()
                if change.state is ChangeState.BUILDING and change.repo == repo
            ),
            None,
        )

    def _reap(self) -> None:
        while len(self._changes) > self._max:
            for key, change in self._changes.items():
                if change.state is not ChangeState.BUILDING:
                    del self._changes[key]
                    break
            else:
                self._changes.popitem(last=False)


# --- the build --------------------------------------------------------------

# Called with the built toplevel; returns the host-action id that now carries it.
Propose = Callable[[ConfigChange], Awaitable[str]]


class ConfigChangeBuilder:
    """Builds a resolved change and hands the result to the propose step."""

    def __init__(
        self,
        *,
        runner: Runner = run_command,
        executor: Executor = run_action,
        build_timeout: float = 7200.0,
    ) -> None:
        self._runner = runner
        self._executor = executor
        self._timeout = build_timeout

    # The git and evaluation reads go through the same seam as the build, so a
    # test drives the whole pipeline through one injection rather than three.

    def resolve(
        self, repo: Path, ref: str, *, allowed: Path | None = None
    ) -> tuple[Path, Resolved]:
        return resolve(repo, ref, runner=self._runner, allowed=allowed)

    def check_attr(self, url: str, attr: str) -> None:
        check_attr(url, attr, runner=self._runner)

    async def stream(
        self, change: ConfigChange, propose: Propose
    ) -> AsyncIterator[ConfigBuildEvent]:
        """Build ``change``, then propose the activation of what was built.

        A build failure is TERMINAL here: the record keeps the log tail and no
        proposal is ever created, so a configuration that does not build has no
        route to activation at all - not because a check refuses it, but because
        the thing an approval would act on does not exist.
        """
        url = flake_url(Path(change.resolved.repo), change.resolved)
        # The probe is a full flake EVALUATION - measured 6.4s warm on this host
        # and slower cold - so it runs HERE rather than in the request that
        # started this. The MCP tool's own API timeout is 15s, so probing before
        # returning made the one call an agent always makes on a changed
        # configuration the one most likely to report a timeout for a build that
        # was in fact running.
        try:
            self.check_attr(url, change.attr)
        except ConfigChangeRefused as exc:
            change.state = ChangeState.FAILED
            change.error = str(exc)
            yield ConfigBuildError(detail=change.error)
            return
        argv = build_argv(url, change.attr)
        collected: list[str] = []
        queue: "asyncio.Queue[ConfigBuildEvent]" = asyncio.Queue()

        def sink(stream: str, text: str) -> None:
            collected.append(text)
            # Bounded: a nix build log can run to megabytes, and the operator is
            # watching the live stream anyway.
            if len(collected) > 4000:
                del collected[: len(collected) - 4000]
            queue.put_nowait(ConfigBuildOutput(stream=stream, text=text))

        yield ConfigBuildOutput(stream="stdout", text=f"$ {' '.join(argv)}\n")
        # The executor's sink is synchronous, so the log reaches the consumer
        # through a queue while the build runs as its own task. Yielding the
        # collected output after the fact would turn a live build log into a
        # transcript that arrives when it is no longer useful.
        build = asyncio.ensure_future(
            self._executor(argv, timeout=self._timeout, on_output=sink)
        )
        try:
            while True:
                pending = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait(
                    {pending, build}, return_when=asyncio.FIRST_COMPLETED
                )
                if pending in done:
                    yield pending.result()
                    continue
                pending.cancel()
                break
            while not queue.empty():
                yield queue.get_nowait()
            result = await build
        except (asyncio.CancelledError, GeneratorExit):
            # The operator stopped watching, or cancelled outright. Kill the
            # build; nix keeps whatever it finished in the store, so the next
            # attempt resumes rather than starting over.
            build.cancel()
            change.state = ChangeState.CANCELLED
            change.error = (
                "the build was stopped before it finished, so nothing was built "
                "and nothing was proposed. What nix had already built stays in "
                "the store and the next attempt reuses it."
            )
            raise

        if result.outcome is not Outcome.OK:
            change.state = ChangeState.FAILED
            change.log_tail = "".join(collected)[-MAX_LOG_TAIL:]
            change.error = f"the configuration did not build: {result.reason()}"
            yield ConfigBuildError(detail=change.error)
            return

        toplevel = toplevel_from(result.stdout)
        if not toplevel:
            change.state = ChangeState.FAILED
            change.log_tail = "".join(collected)[-MAX_LOG_TAIL:]
            change.error = (
                "the build reported success but printed no store path, so there "
                "is nothing identifiable to activate"
            )
            yield ConfigBuildError(detail=change.error)
            return

        change.toplevel = toplevel
        try:
            change.action_id = await propose(change)
        except Exception as exc:  # noqa: BLE001 - the refusal is the answer
            change.state = ChangeState.FAILED
            change.error = f"the built configuration was not accepted: {exc}"
            yield ConfigBuildError(detail=change.error)
            return
        change.state = ChangeState.PROPOSED
        yield ConfigBuildDone(change=change)


def render_change(change: ConfigChange) -> str:
    """One configuration change as plain text, for an agent to relay verbatim.

    Everything an operator needs before they open the approval: which revision,
    what its commit says, whether it is merged, what is NOT in the build, and
    where the change is in its life. The closure diff is not here - it belongs to
    the host action's own preview, which is rendered by
    ``host_actions.render_action``.
    """
    resolved = change.resolved
    lines = [
        f"nixos config change {change.id}",
        f"  repo:     {resolved.repo}",
        f"  ref:      {resolved.ref} @ {resolved.rev[:12]}",
        f"  commit:   {resolved.subject or '(no subject)'}",
        f"  host:     nixosConfigurations.{change.attr}",
        f"  state:    {change.state}",
    ]
    if resolved.merged is False:
        lines.append(
            f"  NOTE:     {resolved.rev[:12]} is not in "
            f"{resolved.head_branch or 'the checkout'} yet - merging it back is a "
            "separate act, and until it happens a later change branched from "
            f"{resolved.head_branch or 'the checkout'} will not contain it"
        )
    if resolved.uncommitted:
        shown = ", ".join(resolved.uncommitted[:10])
        more = (
            f" and {len(resolved.uncommitted) - 10} more"
            if len(resolved.uncommitted) > 10
            else ""
        )
        lines.append(
            f"  NOTE:     {len(resolved.uncommitted)} uncommitted file(s) in that "
            f"working tree are NOT in this build ({shown}{more}): the build takes "
            "the tree from the commit, on purpose"
        )
    if change.toplevel:
        lines.append(f"  built:    {change.toplevel}")
    if change.action_id:
        lines.append(f"  action:   {change.action_id} (awaiting the operator)")
    if change.error:
        lines.append(f"  ERROR:    {change.error}")
    if change.log_tail:
        tail = change.log_tail.strip().splitlines()[-20:]
        lines.append("  last lines of the build log:")
        lines.extend(f"    {line}" for line in tail)
    return "\n".join(lines)


__all__ = [
    "ChangeInFlight",
    "ChangeState",
    "ConfigBuildBus",
    "ConfigBuildDone",
    "ConfigBuildError",
    "ConfigBuildEvent",
    "ConfigBuildOutput",
    "ConfigChange",
    "ConfigChangeBuilder",
    "ConfigChangeRefused",
    "ConfigChangeStore",
    "ConfigSupervisor",
    "Resolved",
    "UnknownChange",
    "build_argv",
    "check_attr",
    "config_supervisor",
    "default_attr",
    "flake_url",
    "render_change",
    "resolve",
    "toplevel_from",
]
