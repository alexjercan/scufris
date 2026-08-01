"""The helper's side of a configuration change: what it would run, and what it refuses.

Driven against a `HostdEngine` over the same faked host every host-action test
uses, so the plan, the preview, rollback and apply are described by one machine
rather than four. Covers the two-command plan and its order, the preview being
honest about what it knows and what it will not run, resolving a rollback from a
generation number, and the apply preflight - including the half-applied state a
switch that fails after the profile moved leaves behind.

What is deliberately NOT here is a real activation: it needs root and a real
system profile, so it lives in ``nix/tests/scufris-hostd-vm.nix``. These prove
everything up to the point where the process would be spawned, plus exactly
which argv it would be.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
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

from scufris.host.run import CommandResult, FakeRunner, Outcome, ok_result
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

REV = "3af39d5ec6bd0616a63855f3be34e94a5b5b6291"


SWITCH_RUNNING = ok_result("active\n")


# The same faked host every host-action test uses, so R1, R2 and R3 are described
# by one machine rather than three.
r3_runner = host_runner


r3_files = host_files


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
