"""R3: a reviewed commit becomes the running system, reversibly.

Three layers, because the failures live at different heights:

- the HELPER's plan and preview (what argv, what the operator is shown, what it
  refuses),
- the APP's build pipeline (resolve a ref, build a commit, propose the result),
  driven over HTTP against a real hostd socket like ``test_host_action_api``,
- the REPOSITORY, against a real temporary git repo, to prove the flow cannot
  write to it.

What is deliberately NOT here is a real activation: it needs root and a real
system profile, so it lives in ``nix/tests/scufris-hostd-vm.nix`` where a real
root helper switches a real NixOS machine. The tests here prove everything up to
the point where the process would be spawned, plus exactly which argv it would
be.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import _Helper
from fastapi.testclient import TestClient
from test_host_action_api import (
    ORIGIN,
    _login,
    _settings,
)
from test_host_actions import (
    BUILT_SYSTEM as BUILT,
)
from test_host_actions import (
    NIX,
    host_files,
    host_runner,
)
from test_host_actions import (
    OLD_SYSTEM as OLD,
)
from test_host_actions import (
    RUNNING_SYSTEM as RUNNING,
)

from scufris.app import create_app
from scufris.auth import CSRF_HEADER
from scufris.host.run import CommandResult, FakeRunner, Outcome, ok_result
from scufris.hostconfig import (
    ConfigChangeRefused,
    build_argv,
    flake_url,
    resolve,
    toplevel_from,
)
from scufris.hostd import (
    ActionKind,
    ActionRefused,
    AuditEvent,
    AuditLog,
    FakeExecutor,
    FakeFiles,
    HostdEngine,
    HostdRefusal,
    ProposalState,
    RiskClass,
    build_plan,
)
from scufris.hostd.actions import SWITCH_UNIT, validate_toplevel
from scufris.hostd.nixos import closure_diff, switch_in_flight
from scufris.hostd.preview import PreviewKind
from scufris.metrics import Collector

REV = "3af39d5ec6bd0616a63855f3be34e94a5b5b6291"

SWITCH_RUNNING = ok_result("active\n")

# The same faked host every host-action test uses, so R1, R2 and R3 are described
# by one machine rather than three.
r3_runner = host_runner
r3_files = host_files

# What the host looks like AFTER an activation: a new generation, and
# /run/current-system pointing at what was built.
GENERATIONS_AFTER_SWITCH = ok_result(
    json.dumps(
        [
            {
                "generation": 192,
                "date": "2026-07-29 12:30:00",
                "nixosVersion": "26.11",
                "current": True,
            },
            {
                "generation": 191,
                "date": "2026-07-29 10:00:00",
                "nixosVersion": "26.11",
                "current": False,
            },
            {
                "generation": 190,
                "date": "2026-07-28 10:00:00",
                "nixosVersion": "26.11",
                "current": False,
            },
        ]
    )
)


def _switched(host: _Helper) -> None:
    """Move the faked host to where a successful activation would have left it."""
    host.runner.results["nixos-rebuild list-generations"] = GENERATIONS_AFTER_SWITCH
    host.files.links["/run/current-system"] = BUILT
    host.files.links["/nix/var/nix/profiles/system-192-link"] = BUILT


def r3_engine(
    tmp_path: Path,
    *,
    runner: FakeRunner | None = None,
    files: FakeFiles | None = None,
    executor: FakeExecutor | None = None,
) -> tuple[HostdEngine, FakeExecutor, AuditLog]:
    log = AuditLog(tmp_path / "audit.jsonl")
    run = executor or FakeExecutor()
    return (
        HostdEngine(
            log,
            runner=runner or r3_runner(),
            executor=run,
            files=files or r3_files(),
        ),
        run,
        log,
    )


def _requester() -> Any:
    from scufris.hostd import Requester

    return Requester(actor="operator")


# --- the plan: two commands, in one order --------------------------------


def test_an_activation_sets_the_profile_then_switches() -> None:
    """The order is the whole point, and it is asserted as an order.

    Switching first would activate a configuration the profile does not name, so
    a reboot would go back to the old one with no record of why.
    """
    plan = build_plan(
        ActionKind.ACTIVATE,
        {"toplevel": BUILT, "repo": "/home/alex/personal/nix.dotfiles", "rev": REV},
        runner=r3_runner(),
        files=r3_files(),
    )

    assert plan.risk is RiskClass.R3
    assert [step.argv for step in plan.steps] == [
        ["nix-env", "--profile", "/nix/var/nix/profiles/system", "--set", BUILT],
        [
            "systemd-run",
            "--collect",
            "--no-ask-password",
            "--pipe",
            "--quiet",
            "--service-type=exec",
            f"--unit={SWITCH_UNIT}",
            "--setenv=NIXOS_INSTALL_BOOTLOADER=0",
            "--",
            f"{BUILT}/bin/switch-to-configuration",
            "switch",
        ],
    ]
    # Not a rebuild: nothing here re-evaluates the flake, because what gets
    # activated must be the exact path that was previewed. (The transient unit
    # NAME contains "nixos-rebuild" - that is systemd-run's --unit, deliberately
    # shared with nixos-rebuild so the two cannot switch at once.)
    flat = " ".join(part for step in plan.steps for part in step.argv)
    assert [step.argv[0] for step in plan.steps] == ["nix-env", "systemd-run"]
    assert "--flake" not in flat
    assert "nix build" not in flat
    # And the bootloader is not reinstalled as a side effect of a switch.
    assert "--setenv=NIXOS_INSTALL_BOOTLOADER=0" in plan.steps[1].argv


@pytest.mark.parametrize(
    "candidate,because",
    [
        (f"{BUILT}/bin/switch-to-configuration", "names the ROOT"),
        ("/nix/store/../etc/shadow", "names the ROOT"),
        ("./result", "names the ROOT"),
        ("/nix/store/notahash-thing", "names the ROOT"),
    ],
)
def test_a_toplevel_that_is_not_a_store_root_is_refused(
    candidate: str, because: str
) -> None:
    """A subpath, a traversal and a relative path are all refused by SHAPE."""
    with pytest.raises(ActionRefused) as refused:
        validate_toplevel(candidate, runner=r3_runner(), files=r3_files())
    assert because in str(refused.value)


def test_a_path_this_store_does_not_have_is_refused() -> None:
    unknown = CommandResult(
        argv=[], outcome=Outcome.FAILED, stderr="path does not exist", returncode=1
    )
    with pytest.raises(ActionRefused) as refused:
        validate_toplevel(
            BUILT, runner=r3_runner(**{f"{NIX} path-info": unknown}), files=r3_files()
        )
    assert "does not have it as a valid path" in str(refused.value)


def test_a_store_path_that_is_not_a_nixos_system_is_refused() -> None:
    """nixos-rebuild's own precondition, for the reason nixos-rebuild has it."""
    files = r3_files()
    files.files.discard(f"{BUILT}/nixos-version")
    with pytest.raises(ActionRefused) as refused:
        validate_toplevel(BUILT, runner=r3_runner(), files=files)
    assert "no nixos-version" in str(refused.value)

    files = r3_files()
    files.executables.discard(f"{BUILT}/bin/switch-to-configuration")
    with pytest.raises(ActionRefused) as refused:
        validate_toplevel(BUILT, runner=r3_runner(), files=files)
    assert "no bin/switch-to-configuration" in str(refused.value)


# --- the preview: honest about what it knows and what it will not run ----


@pytest.mark.asyncio
async def test_the_preview_never_runs_the_proposed_configuration(
    tmp_path: Path,
) -> None:
    """Producing a unit list would mean executing an unapproved config as root.

    This is the one preview the framework deliberately narrows, so it is pinned
    by BOTH halves: nothing from the proposed toplevel is executed, and the
    preview says so in words rather than leaving a gap.
    """
    core, executor, _log = r3_engine(tmp_path)
    runner = r3_runner()
    core._runner = runner  # type: ignore[attr-defined]

    view = await core.propose(
        ActionKind.ACTIVATE, {"toplevel": BUILT, "rev": REV}, _requester()
    )

    ran = [" ".join(call) for call in runner.calls]
    assert executor.calls == []
    assert not [call for call in ran if "switch-to-configuration" in call]
    assert not [call for call in ran if "dry-activate" in call]
    body = "\n".join(view.preview.lines)
    assert "units that would restart is NOT shown" in body
    assert "would defeat the approval" in body


@pytest.mark.asyncio
async def test_no_closure_change_is_stated_rather_than_shown_as_nothing(
    tmp_path: Path,
) -> None:
    """The measured trap: identical closures print NOTHING and exit 0.

    So the assertion that matters is the ABSENCE of the misleading rendering -
    an empty diff section that reads the same as a preview whose command failed.
    """
    core, _executor, _log = r3_engine(
        tmp_path, runner=r3_runner(**{f"{NIX} store diff-closures": ok_result("")})
    )

    view = await core.propose(ActionKind.ACTIVATE, {"toplevel": BUILT}, _requester())

    body = "\n".join(view.preview.lines)
    assert "no closure change" in body
    assert "byte-identical" in body
    # The failure rendering must not be reachable from a successful empty diff,
    # and the section header that would leave a blank list must be absent.
    assert view.preview.ok
    assert view.preview.kind is PreviewKind.SIMULATION
    assert "closure diff (what packages change):" not in body
    assert "could not" not in body


@pytest.mark.asyncio
async def test_a_preview_whose_diff_failed_is_refused_not_shown_empty(
    tmp_path: Path,
) -> None:
    """The other half of the same trap: a broken diff is not "no change"."""
    broken = CommandResult(
        argv=[], outcome=Outcome.FAILED, stderr="error: opening store", returncode=1
    )
    core, _executor, _log = r3_engine(
        tmp_path, runner=r3_runner(**{f"{NIX} store diff-closures": broken})
    )

    with pytest.raises(HostdRefusal) as refused:
        await core.propose(ActionKind.ACTIVATE, {"toplevel": BUILT}, _requester())

    assert "no preview could be produced" in refused.value.detail
    assert "not approvable" in refused.value.detail


def test_a_closure_diff_is_stripped_of_colour_and_non_ascii() -> None:
    """Measured: NO_COLOR does not silence nix, so this module has to."""
    lines, changed, caveat = closure_diff(r3_runner(), RUNNING, BUILT)

    assert changed and not caveat
    body = "\n".join(lines)
    assert "\x1b[" not in body
    assert body.isascii(), body
    assert "acl: 2.3.2 -> 2.4.0" in body
    assert "ripgrep: (none) -> 14.1.1" in body


@pytest.mark.asyncio
async def test_an_activation_offers_a_rollback_to_what_is_running(
    tmp_path: Path,
) -> None:
    core, _executor, _log = r3_engine(tmp_path)

    view = await core.propose(ActionKind.ACTIVATE, {"toplevel": BUILT}, _requester())

    assert view.reversal.possible
    assert view.reversal.kind is ActionKind.ROLLBACK
    assert view.reversal.args == {"generation": 191}
    assert "191" in view.reversal.summary


# --- rollback: a number, and the helper resolves the rest ----------------


def test_a_rollback_resolves_its_own_toplevel() -> None:
    """The caller names a generation; the profile decides what that means."""
    plan = build_plan(
        ActionKind.ROLLBACK, {"generation": 190}, runner=r3_runner(), files=r3_files()
    )

    assert plan.steps[0].argv == [
        "nix-env",
        "--profile",
        "/nix/var/nix/profiles/system",
        "--switch-generation",
        "190",
    ]
    # Resolved from /nix/var/nix/profiles/system-190-link, not supplied.
    assert plan.steps[1].argv[-2] == f"{OLD}/bin/switch-to-configuration"
    assert plan.args["toplevel"] == OLD


def test_a_rollback_to_the_current_generation_is_refused() -> None:
    with pytest.raises(ActionRefused) as refused:
        build_plan(
            ActionKind.ROLLBACK,
            {"generation": 191},
            runner=r3_runner(),
            files=r3_files(),
        )
    assert "already running" in str(refused.value)


def test_a_rollback_to_a_generation_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ActionRefused) as refused:
        build_plan(
            ActionKind.ROLLBACK,
            {"generation": 999},
            runner=r3_runner(),
            files=r3_files(),
        )
    assert "no generation 999" in str(refused.value)
    # And it says which ones there are, rather than just "no".
    assert "191" in str(refused.value)


def test_a_rollback_whose_generation_link_is_gone_is_refused() -> None:
    """Listed but unresolvable: a collected generation must not become a guess."""
    files = r3_files()
    del files.links["/nix/var/nix/profiles/system-190-link"]
    with pytest.raises(ActionRefused) as refused:
        build_plan(
            ActionKind.ROLLBACK, {"generation": 190}, runner=r3_runner(), files=files
        )
    assert "does not resolve" in str(refused.value)


# --- apply: the preflight, the order, and the half-applied state ---------


@pytest.mark.asyncio
async def test_a_second_switch_is_refused_before_the_profile_moves(
    tmp_path: Path,
) -> None:
    """Two interleaved activations leave a system that matches neither.

    The refusal has to land BEFORE the first step: a profile pointed at a new
    configuration while someone else's switch is running is the split state this
    check exists to avoid.
    """
    core, executor, log = r3_engine(tmp_path)
    view = await core.propose(ActionKind.ACTIVATE, {"toplevel": BUILT}, _requester())
    core._runner = r3_runner(  # type: ignore[attr-defined]
        **{"systemctl is-active": SWITCH_RUNNING}
    )

    with pytest.raises(HostdRefusal) as refused:
        await core.apply(view.id, on_output=lambda *_: None, approved_by="operator")

    assert "already running" in refused.value.detail
    assert executor.calls == [], "the profile must not move"
    denied = [r for r in log.tail(50) if r.event is AuditEvent.DENIED]
    assert denied and denied[-1].outcome == "blocked"


@pytest.mark.asyncio
async def test_an_activation_that_cannot_ask_about_a_running_switch_refuses(
    tmp_path: Path,
) -> None:
    """ "Probably not" is not good enough for the one act with no halfway point."""
    missing = CommandResult(argv=[], outcome=Outcome.MISSING)
    assert "cannot tell" in switch_in_flight(
        r3_runner(**{"systemctl is-active": missing})
    )


@pytest.mark.asyncio
async def test_a_switch_that_fails_after_the_profile_moved_is_recorded_as_split(
    tmp_path: Path,
) -> None:
    """The failure that is not "nothing happened".

    Step 1 succeeded, so the NEXT boot runs the new configuration while THIS one
    does not. A record that said only "failed" would send the operator to bed
    with a machine that boots into something nobody approved as running.
    """

    class _SecondStepFails(FakeExecutor):
        async def __call__(  # type: ignore[override]
            self, argv: list[str], *, timeout: float, on_output: Any
        ) -> CommandResult:
            self.calls.append(list(argv))
            if "switch-to-configuration" in " ".join(argv):
                return CommandResult(
                    argv=argv,
                    outcome=Outcome.FAILED,
                    stderr="activation script failed",
                    returncode=1,
                )
            return CommandResult(argv=argv, outcome=Outcome.OK, returncode=0)

    core, executor, log = r3_engine(tmp_path, executor=_SecondStepFails())
    view = await core.propose(ActionKind.ACTIVATE, {"toplevel": BUILT}, _requester())

    result = await core.apply(
        view.id, on_output=lambda *_: None, approved_by="operator"
    )

    assert not result.ok
    assert result.steps_completed == 1 and result.steps_total == 2
    assert "step 2 of 2 failed" in result.detail
    assert "NEXT boot" in result.detail
    assert "roll back" in result.detail
    failed = [r for r in log.tail(50) if r.event is AuditEvent.FAILED]
    assert failed and "NEXT boot" in failed[-1].detail
    assert failed[-1].steps_completed == 1
    # Both steps were attempted in order, and only in that order.
    assert len(executor.calls) == 2
    assert executor.calls[0][:2] == ["nix-env", "--profile"]


@pytest.mark.asyncio
async def test_a_drifted_system_refuses_the_approval(tmp_path: Path) -> None:
    """Someone switched between the preview and the approval."""
    core, executor, _log = r3_engine(tmp_path)
    view = await core.propose(ActionKind.ACTIVATE, {"toplevel": BUILT}, _requester())
    # The machine is now running something else, on a new generation.
    moved = r3_files()
    moved.links["/run/current-system"] = OLD
    core._files = moved  # type: ignore[attr-defined]

    with pytest.raises(HostdRefusal) as refused:
        await core.apply(view.id, on_output=lambda *_: None, approved_by="operator")

    assert refused.value.code.value == "drifted"
    assert executor.calls == []
    assert core.proposal(view.id) is not None
    assert core.proposal(view.id).state is ProposalState.DRIFTED  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_cancelled_activation_says_the_switch_was_not_stopped(
    tmp_path: Path,
) -> None:
    """Honesty about what cancel achieves: it stops watching, not switching."""
    core, _executor, log = r3_engine(tmp_path, executor=FakeExecutor(hang=True))
    view = await core.propose(ActionKind.ACTIVATE, {"toplevel": BUILT}, _requester())
    task = asyncio.ensure_future(
        core.apply(view.id, on_output=lambda *_: None, approved_by="operator")
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    cancelled = [r for r in log.tail(50) if r.event is AuditEvent.CANCELLED]
    assert cancelled
    detail = cancelled[-1].detail
    assert SWITCH_UNIT in detail
    assert "stops WATCHING it, not the activation itself" in detail


# --- the app: build a commit, then propose it ----------------------------


class _BuildExecutor(FakeExecutor):
    """A build that prints a store path, and then whatever the apply needs."""

    def __init__(self, *, out: str = BUILT, fail: bool = False, hang: bool = False):
        super().__init__()
        self.out = out
        self.fail = fail
        self.hang_build = hang

    async def __call__(  # type: ignore[override]
        self, argv: list[str], *, timeout: float, on_output: Any
    ) -> CommandResult:
        self.calls.append(list(argv))
        if argv[0] == "nix" and "build" in argv:
            on_output("stderr", "building nixos-system...\n")
            self.started.set()
            if self.hang_build:
                await asyncio.sleep(3600)
            if self.fail:
                # run_action streams stderr through on_output as it arrives, so a
                # fake that only returned it would exercise a path the real
                # executor does not have.
                on_output("stderr", "error: attribute 'ripgrp' missing\n")
                return CommandResult(
                    argv=argv,
                    outcome=Outcome.FAILED,
                    stdout="",
                    stderr="error: attribute 'ripgrp' missing\n",
                    returncode=1,
                )
            return CommandResult(
                argv=argv, outcome=Outcome.OK, stdout=f"{self.out}\n", returncode=0
            )
        return CommandResult(argv=argv, outcome=Outcome.OK, returncode=0)


@pytest.fixture
def config_repo(tmp_path: Path) -> Path:
    """A real git repository with a flake, committed on a branch."""
    if shutil.which("git") is None:  # pragma: no cover - git is in the dev shell
        pytest.skip("git is not on PATH")
    repo = tmp_path / "nix.dotfiles"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            env={
                # PATH from the environment, not a guess: `nix flake check` runs
                # pytest in a sandbox where /usr/bin does not exist and git comes
                # from the check's own nativeBuildInputs. HOME is still
                # overridden, which is what keeps the repo hermetic.
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(tmp_path),
                "GIT_AUTHOR_NAME": "t",
                "GIT_AUTHOR_EMAIL": "t@e",
                "GIT_COMMITTER_NAME": "t",
                "GIT_COMMITTER_EMAIL": "t@e",
            },
        ).stdout.strip()

    git("init", "-q", "-b", "master")
    (repo / "flake.nix").write_text("{ outputs = _: {}; }\n")
    git("add", "flake.nix")
    git("commit", "-qm", "initial")
    git("checkout", "-qb", "config/add-ripgrep")
    (repo / "packages.nix").write_text("[ ripgrep ]\n")
    git("add", "packages.nix")
    git("commit", "-qm", "feat: add ripgrep")
    return repo


def _repo_state(repo: Path) -> tuple[str, str, str]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True
        ).stdout

    return (
        git("status", "--porcelain"),
        git("log", "--all", "--format=%H"),
        git("branch", "--format=%(refname)"),
    )


def test_nixos_change_never_writes_to_the_config_repo(config_repo: Path) -> None:
    """Cleanliness is structural here, not something a teardown achieves.

    The build addresses the repository as `git+file://...?rev=`, so nix reads the
    tree from the commit: there is no worktree to leave behind, no `result`
    symlink, no lock-file write and no commit. This replaces the planned
    `test_rejected_nixos_proposal_leaves_repo_clean` - with the edit owned by the
    project flow, there is nothing for a rejected proposal to clean up.
    """
    before = _repo_state(config_repo)

    main, resolved = resolve(config_repo, "config/add-ripgrep")
    url = flake_url(main, resolved)
    argv = build_argv(url, "nixos")

    assert main == config_repo
    assert resolved.rev and resolved.subject == "feat: add ripgrep"
    # The revision is pinned INTO the flake reference, so the working tree is not
    # what gets built.
    assert f"rev={resolved.rev}" in url
    assert "ref=config/add-ripgrep" in url
    assert argv[0] == "nix" and "build" in argv
    for flag in ("--no-link", "--no-update-lock-file", "--no-write-lock-file"):
        assert flag in argv
    # Nothing in the argv names the working tree as a source.
    assert not [part for part in argv if part == str(config_repo)]

    assert _repo_state(config_repo) == before
    assert not (config_repo / "result").exists()
    # And no worktree was created anywhere for this.
    listed = subprocess.run(
        ["git", "-C", str(config_repo), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
    ).stdout
    assert listed.count("worktree ") == 1


def test_an_uncommitted_edit_is_reported_as_not_in_the_build(
    config_repo: Path,
) -> None:
    """An agent that edited but did not commit must be told, not left guessing."""
    (config_repo / "packages.nix").write_text("[ ripgrep fd ]\n")

    _main, resolved = resolve(config_repo, "HEAD")

    assert resolved.uncommitted == ["packages.nix"]
    from scufris.hostconfig import ConfigChange, render_change

    text = render_change(ConfigChange(id="x", resolved=resolved, attr="nixos"))
    assert "are NOT in this build" in text
    assert "packages.nix" in text


def test_a_ref_that_does_not_exist_is_refused_by_name(config_repo: Path) -> None:
    with pytest.raises(ConfigChangeRefused) as refused:
        resolve(config_repo, "config/typo")
    assert "does not name a commit" in str(refused.value)


@pytest.mark.parametrize("hostile", ["-c", "--upload-pack=x", "a..b", "master;rm"])
def test_a_ref_outside_the_charset_is_refused(config_repo: Path, hostile: str) -> None:
    """A ref reaches a git argv, so it is charset-validated like a unit name."""
    with pytest.raises(ConfigChangeRefused):
        resolve(config_repo, hostile)


def test_a_repository_other_than_this_host_s_configuration_is_refused(
    config_repo: Path, tmp_path: Path
) -> None:
    """Which revision to build is a caller's choice; which repository is not.

    Without this, an agent could commit its own flake anywhere it can write and
    have the server build and propose THAT as the system - the same shape as
    handing over a store path, one step removed.
    """
    other = tmp_path / "mine"
    other.mkdir()
    subprocess.run(["git", "-C", str(other), "init", "-q"], check=True)
    (other / "flake.nix").write_text("{ outputs = _: {}; }\n")

    with pytest.raises(ConfigChangeRefused) as refused:
        resolve(other, "HEAD", allowed=config_repo)
    assert "is not it" in str(refused.value)

    # A WORKTREE of the allowed repository is fine - that is where an agent
    # works - because the check is on the main repository the commits live in.
    worktree = tmp_path / "wt"
    subprocess.run(
        ["git", "-C", str(config_repo), "worktree", "add", "-q", str(worktree)],
        check=True,
        capture_output=True,
    )
    main, resolved = resolve(worktree, "HEAD", allowed=config_repo)
    assert main == config_repo
    assert resolved.rev


def test_a_ref_of_head_is_recorded_as_the_branch_it_is(config_repo: Path) -> None:
    """ "ref: HEAD @ 3af39d5" in an approval prompt tells the operator nothing."""
    _main, resolved = resolve(config_repo, "HEAD")

    assert resolved.ref == "config/add-ripgrep"
    assert "ref=config/add-ripgrep" in flake_url(config_repo, resolved)


def test_the_attribute_probe_does_not_delay_the_request(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """An unknown host attribute fails the CHANGE, not the request that started it.

    The probe is a full flake evaluation (measured 6.4s warm, slower cold) and
    the MCP tool's API timeout is 15s, so probing before answering made the one
    call an agent always makes report a timeout for a build that was running.
    """
    app = _app(tmp_path, fake_collector, helper, config_repo)
    client = make_client(app)
    csrf = _login(client)

    resp = _post(
        client,
        csrf,
        "/api/host/config/changes",
        ref="config/add-ripgrep",
        attr="not-this-host",
    )

    assert resp.status_code == 201, resp.text
    change = _settle(client, csrf, resp.json()["id"], want="failed")
    assert "no nixosConfigurations.not-this-host" in change["error"]
    # It says which ones exist, and it never reached a build.
    assert "nixos" in change["error"]
    assert change["action_id"] == ""


def test_a_directory_that_is_not_a_flake_is_refused(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ConfigChangeRefused):
        resolve(plain, "HEAD")


def test_only_a_store_path_is_taken_from_a_build() -> None:
    """The out path is read from stdout, and only if it IS a store path."""
    assert toplevel_from(f"warning: dirty tree\n{BUILT}\n") == BUILT
    assert toplevel_from("built nothing\n") == ""
    assert toplevel_from("/etc/passwd\n") == ""


# --- the app over HTTP ----------------------------------------------------


def _app(
    tmp_path: Path,
    fake_collector: Collector,
    host: _Helper,
    config_repo: Path,
    *,
    build: _BuildExecutor | None = None,
) -> Any:
    from scufris.hostconfig import ConfigChangeBuilder

    return create_app(
        collector=fake_collector,
        settings=_settings(
            tmp_path,
            host,
            host_config_repo=config_repo,
            host_config_attr="nixos",
        ),
        # The attribute probe and the git reads go through the runner; the BUILD
        # goes through the executor, which is the only thing being faked.
        config_builder=ConfigChangeBuilder(
            runner=_config_runner(),
            executor=build or _BuildExecutor(),
        ),
    )


def _config_runner(**overrides: CommandResult) -> FakeRunner:
    """git reads run for real; only the flake evaluation is canned."""
    results: dict[str, CommandResult] = {
        f"{NIX} eval": ok_result(json.dumps(["nixos"])),
    }
    results.update(overrides)
    real = FakeRunner(results=results)

    def run(argv: list[str], *, timeout: float = 30.0) -> CommandResult:
        if argv and argv[0] == "git":
            from scufris.host.run import run_command

            return run_command(argv, timeout=timeout)
        return real(argv, timeout=timeout)

    return run  # type: ignore[return-value]


def _post(client: TestClient, csrf: str, path: str, **body: Any) -> Any:
    return client.post(path, json=body, headers={"Origin": ORIGIN, CSRF_HEADER: csrf})


def _settle(client: TestClient, csrf: str, change_id: str, *, want: str) -> Any:
    """Poll a change until it leaves `building`."""
    for _ in range(200):
        change = client.get(f"/api/host/config/changes/{change_id}").json()
        if change["state"] != "building":
            assert change["state"] == want, change
            return change
        import time as _time

        _time.sleep(0.05)
    raise AssertionError("the change never settled")


def test_propose_refuses_a_caller_supplied_toplevel(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """If the caller picks the store path, the closure diff describes its choice.

    So the generic propose surface has no `activate` at all, and the refusal
    names the endpoint that does build one.
    """
    client = make_client(_app(tmp_path, fake_collector, helper, config_repo))
    csrf = _login(client)

    resp = _post(
        client,
        csrf,
        "/api/host/actions",
        kind="activate",
        args={"toplevel": BUILT},
    )

    assert resp.status_code == 422, resp.text
    assert "not proposed directly" in resp.text
    assert "/api/host/config/changes" in resp.text
    # Nothing reached the helper, so there is no proposal to approve.
    assert client.get("/api/host/actions").json() == []
    assert helper.executor.calls == []


def test_nixos_build_failure_blocks_activation(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """A configuration that does not build has no route to activation.

    Not because a check refuses it: because the thing an approval would act on
    does not exist. The log comes back so the failure is diagnosable.
    """
    app = _app(
        tmp_path, fake_collector, helper, config_repo, build=_BuildExecutor(fail=True)
    )
    client = make_client(app)
    csrf = _login(client)

    resp = _post(client, csrf, "/api/host/config/changes", ref="config/add-ripgrep")
    assert resp.status_code == 201, resp.text
    change = _settle(client, csrf, resp.json()["id"], want="failed")

    assert change["action_id"] == ""
    assert "did not build" in change["error"]
    assert "ripgrp" in change["log_tail"]
    # No proposal exists, so there is nothing an approval could act on - and the
    # helper never ran anything.
    assert client.get("/api/host/actions").json() == []
    assert helper.executor.calls == []


def test_nixos_change_builds_diffs_switches_and_rolls_back(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """The whole contract, end to end, over HTTP against the real helper.

    resolve a ref -> build the commit -> a closure-diff preview -> the operator
    approves -> the profile moves and the system switches -> the generation it
    replaced is recorded as the way back -> roll back to it.
    """
    app = _app(tmp_path, fake_collector, helper, config_repo)
    client = make_client(app)
    csrf = _login(client)

    resp = _post(client, csrf, "/api/host/config/changes", ref="config/add-ripgrep")
    assert resp.status_code == 201, resp.text
    change = _settle(client, csrf, resp.json()["id"], want="proposed")

    # The build was addressed at the commit, and it produced a real store path.
    assert change["resolved"]["rev"]
    assert change["toplevel"] == BUILT
    assert change["action_id"]

    action = client.get(f"/api/host/actions/{change['action_id']}").json()
    assert action["proposal"]["risk"] == "r3"
    assert action["proposal"]["kind"] == "activate"
    body = "\n".join(action["proposal"]["preview"]["lines"])
    assert "closure diff" in body
    assert "ripgrep: (none) -> 14.1.1" in body
    assert change["resolved"]["rev"] in body
    # Approving is what runs it. Until then, nothing has.
    assert helper.executor.calls == []

    approved = _post(client, csrf, f"/api/host/actions/{change['action_id']}/approve")
    assert approved.status_code == 200, approved.text

    for _ in range(200):
        action = client.get(f"/api/host/actions/{change['action_id']}").json()
        if action["result"] is not None:
            break
        import time as _time

        _time.sleep(0.05)
    assert action["result"] is not None and action["result"]["ok"], action
    assert action["result"]["steps_completed"] == 2

    # The profile moved first, and then THAT path was switched to.
    assert helper.executor.calls[0] == [
        "nix-env",
        "--profile",
        "/nix/var/nix/profiles/system",
        "--set",
        BUILT,
    ]
    assert helper.executor.calls[1][-2:] == [
        f"{BUILT}/bin/switch-to-configuration",
        "switch",
    ]

    # The way back was recorded at approval time, as a generation number.
    assert action["proposal"]["reversal"]["possible"]
    assert action["proposal"]["reversal"]["args"] == {"generation": 191}

    # The machine has now switched, so the faked host moves with it: generation
    # 192 is current and /run/current-system points at what was built. A rollback
    # proposed against a host still claiming to run 191 would be testing a
    # machine that cannot exist.
    _switched(helper)

    reverted = _post(client, csrf, f"/api/host/actions/{change['action_id']}/revert")
    assert reverted.status_code == 201, reverted.text
    rollback = reverted.json()
    assert rollback["proposal"]["kind"] == "rollback"
    assert rollback["proposal"]["args"]["generation"] == 191
    # The rollback resolved its own store path from the profile.
    assert rollback["proposal"]["args"]["toplevel"] == RUNNING

    # `id` is a property on the record, so the wire shape carries it on the
    # proposal.
    rollback_id = rollback["proposal"]["id"]
    approved = _post(client, csrf, f"/api/host/actions/{rollback_id}/approve")
    assert approved.status_code == 200, approved.text
    for _ in range(200):
        rollback = client.get(f"/api/host/actions/{rollback_id}").json()
        if rollback["result"] is not None:
            break
        import time as _time

        _time.sleep(0.05)
    assert rollback["result"] is not None and rollback["result"]["ok"], rollback
    assert helper.executor.calls[2] == [
        "nix-env",
        "--profile",
        "/nix/var/nix/profiles/system",
        "--switch-generation",
        "191",
    ]

    # And the helper's own record carries the whole story, including the
    # revision that produced what is now running.
    audit = client.get("/api/host/audit?limit=100").json()
    applied = [r for r in audit if r["event"] == "applied"]
    assert [r["kind"] for r in applied] == ["activate", "rollback"]
    assert applied[0]["args"]["rev"] == change["resolved"]["rev"]
    assert applied[0]["args"]["toplevel"] == BUILT
    assert applied[0]["restore_point"]
    assert "191" in applied[0]["reversal"]


def test_concurrent_nixos_proposals_are_serialized(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """One build per repository, refused rather than quietly queued.

    A queued NixOS build sits for an hour with no visible reason, and two builds
    of one repository contend for the same evaluation and store.
    """
    app = _app(
        tmp_path, fake_collector, helper, config_repo, build=_BuildExecutor(hang=True)
    )
    client = make_client(app)
    csrf = _login(client)

    first = _post(client, csrf, "/api/host/config/changes", ref="config/add-ripgrep")
    assert first.status_code == 201, first.text
    second = _post(client, csrf, "/api/host/config/changes", ref="master")

    assert second.status_code == 409, second.text
    assert first.json()["id"] in second.text
    assert "already running" in second.text
    # The first one is still the only change, and still building.
    listed = client.get("/api/host/config/changes").json()
    assert [c["id"] for c in listed] == [first.json()["id"]]
    assert listed[0]["state"] == "building"

    # Cancelling it is not an operator-only act - a build holds no privilege -
    # and it leaves no proposal behind.
    stopped = _post(
        client, csrf, f"/api/host/config/changes/{first.json()['id']}/cancel"
    )
    assert stopped.status_code == 200, stopped.text
    change = _settle(client, csrf, first.json()["id"], want="cancelled")
    assert change["action_id"] == ""
    assert client.get("/api/host/actions").json() == []
