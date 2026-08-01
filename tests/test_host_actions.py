"""The host action framework, driven the way the app drives it.

These are adversarial by design. The framework's whole claim is that a mutating
host action cannot reach the system without a preview and an approval, so the
tests that matter are the ones that try to get around that: a forged id, a
replayed approval, an approval after the system moved, two approvals racing, a
cancellation mid-apply, and an argument that would turn into a flag.

Nothing here spawns a process or touches the host: the engine takes an injected
``Runner`` (canned command output) and an injected ``Executor`` (a scripted
apply), which is what makes the cancellation path testable at all.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from scufris.host.run import (
    NIX_FEATURES,
    CommandResult,
    FakeRunner,
    Outcome,
    ok_result,
)
from scufris.hostd import (
    ActionKind,
    ActionRefused,
    AuditEvent,
    AuditLog,
    ErrorCode,
    FakeExecutor,
    FakeFiles,
    HostdEngine,
    HostdRefusal,
    ProposalState,
    build_plan,
    normalise_unit,
)
from scufris.hostd.actions import PROTECTED_GENERATIONS, generations_older_than
from scufris.hostd.preview import PreviewKind

NOW = datetime(2026, 7, 29, 12, 0, 0)

# The prefix a new-CLI nix invocation actually has: the experimental features are
# explicit in the argv, so a FakeRunner key has to include them (see
# `host.run.nix_cli` for why they are there at all).
NIX = " ".join(["nix", *NIX_FEATURES])

UNIT_SHOW = ok_result(
    "Id=nginx.service\n"
    "Description=Nginx Web Server\n"
    "LoadState=loaded\n"
    "ActiveState=active\n"
    "SubState=running\n"
    "UnitFileState=enabled\n"
    "MainPID=4242\n"
    "Result=success\n"
    "NRestarts=0\n"
)

UNIT_SHOW_INACTIVE = ok_result(
    "Id=nginx.service\n"
    "Description=Nginx Web Server\n"
    "LoadState=loaded\n"
    "ActiveState=inactive\n"
    "SubState=dead\n"
    "UnitFileState=enabled\n"
)

REVERSE_DEPS = ok_result("nginx.service\nmulti-user.target\n")

GENERATIONS = ok_result(
    json.dumps(
        [
            {
                "generation": 191,
                "date": "2026-07-29 10:00:00",
                "nixosVersion": "26.11",
                "current": True,
            },
            {
                "generation": 190,
                "date": "2026-07-28 10:00:00",
                "nixosVersion": "26.11",
                "current": False,
            },
            {
                "generation": 180,
                "date": "2026-01-01 10:00:00",
                "nixosVersion": "26.05",
                "current": False,
            },
            {
                "generation": 179,
                "date": "2025-12-01 10:00:00",
                "nixosVersion": "26.05",
                "current": False,
            },
        ]
    )
)

DEAD_PATHS = ok_result("/nix/store/aaa-thing\n/nix/store/bbb-other\n")

PATH_INFO = ok_result(
    json.dumps(
        {
            "/nix/store/aaa-thing": {"narSize": 1024},
            "/nix/store/bbb-other": {"narSize": 2048},
        }
    )
)

DRY_RUN = ok_result("7642 store paths would be deleted\n")

# --- R3 shapes, shared with tests/test_nixos_config_change.py -------------
#
# Real shapes: a 32-character nix base-32 hash, and a closure diff that still
# carries the colour codes and non-ASCII glyphs nix emits into a pipe (measured
# on this host - NO_COLOR does not silence them).
RUNNING_SYSTEM = "/nix/store/bnfi69bsjhaj4jgp42jk9ys6y80pb9qh-nixos-system-nixos-26.11"
BUILT_SYSTEM = "/nix/store/c0z2q4wl5m7dnpx9rsv0abcdfghijklm-nixos-system-nixos-26.11"
OLD_SYSTEM = "/nix/store/d1y3r5xm6n8fpq0svw1bcdfghijklmnp-nixos-system-nixos-26.05"

CLOSURE_DIFF = ok_result(
    "acl: 2.3.2 \u2192 2.4.0, \x1b[31;1m30.4 KiB\x1b[0m\n"
    "bootspec: 2.0.0 \u2192 \u2205, \x1b[32;1m-1.9 MiB\x1b[0m\n"
    "ripgrep: \u2205 \u2192 14.1.1, \x1b[31;1m4.1 MiB\x1b[0m\n"
)

# `systemctl is-active` on a transient unit that is not running: "inactive",
# exit 3.
NO_SWITCH_RUNNING = CommandResult(
    argv=[], outcome=Outcome.FAILED, stdout="inactive\n", returncode=3
)


def host_files(**links: str) -> FakeFiles:
    """A filesystem where the running system and both toplevels look real."""
    files = FakeFiles(
        files={
            f"{path}/nixos-version"
            for path in (RUNNING_SYSTEM, BUILT_SYSTEM, OLD_SYSTEM)
        },
        executables={
            f"{path}/bin/switch-to-configuration"
            for path in (RUNNING_SYSTEM, BUILT_SYSTEM, OLD_SYSTEM)
        },
        links={
            "/run/current-system": RUNNING_SYSTEM,
            "/nix/var/nix/profiles/system-191-link": RUNNING_SYSTEM,
            "/nix/var/nix/profiles/system-190-link": OLD_SYSTEM,
        },
    )
    files.links.update(links)
    return files


def host_runner(**overrides: CommandResult) -> FakeRunner:
    """A runner replaying this host's real command shapes."""
    results = {
        "systemctl --system show": UNIT_SHOW,
        "systemctl list-dependencies": REVERSE_DEPS,
        "nixos-rebuild list-generations": GENERATIONS,
        "nix-store --gc --print-dead": DEAD_PATHS,
        f"{NIX} path-info": PATH_INFO,
        "nix-collect-garbage --dry-run": DRY_RUN,
        # R3
        f"{NIX} store diff-closures": CLOSURE_DIFF,
        "systemctl is-active": NO_SWITCH_RUNNING,
    }
    results.update(overrides)
    return FakeRunner(results=results)


def engine(
    tmp_path: Path,
    *,
    runner: FakeRunner | None = None,
    executor: FakeExecutor | None = None,
    files: FakeFiles | None = None,
    ttl: float = 600.0,
    clock: object = None,
) -> tuple[HostdEngine, FakeExecutor, AuditLog]:
    log = AuditLog(tmp_path / "audit.jsonl", secrets=frozenset({"s3cr3t-value"}))
    run = executor or FakeExecutor()
    kwargs = {} if clock is None else {"clock": clock}
    return (
        HostdEngine(
            log,
            runner=runner or host_runner(),
            executor=run,
            files=files or host_files(),
            ttl_seconds=ttl,
            **kwargs,  # type: ignore[arg-type]
        ),
        run,
        log,
    )


# --- the central contract ------------------------------------------------


@pytest.mark.asyncio
async def test_host_action_requires_preview_and_approval(tmp_path: Path) -> None:
    """Proposing runs nothing, and only an id the helper issued can be applied.

    The two halves of "no action without a preview and an approval": proposing
    is inert, and there is no way to name a command - only a proposal id, which
    only the helper mints.
    """
    core, executor, log = engine(tmp_path)

    view = await core.propose(ActionKind.UNIT_RESTART, {"unit": "nginx"}, _requester())

    # Proposing previewed it and did not run it.
    assert executor.calls == []
    assert view.state is ProposalState.PENDING
    assert view.steps[0].argv == ["systemctl", "restart", "--", "nginx.service"]
    assert view.preview.kind is PreviewKind.STATE
    assert "not a prediction" in view.preview.label

    # An id the helper never issued cannot be applied, however well-formed.
    with pytest.raises(HostdRefusal) as forged:
        await core.apply("f" * 32, on_output=_ignore)
    assert forged.value.code is ErrorCode.NOT_FOUND
    assert executor.calls == []

    # The real id runs exactly the argv that was previewed.
    result = await core.apply(view.id, on_output=_ignore, approved_by="operator")
    assert result.ok
    assert executor.calls == [["systemctl", "restart", "--", "nginx.service"]]

    events = [record.event for record in log.tail(20)]
    assert AuditEvent.REQUESTED in events
    assert AuditEvent.APPROVED in events
    assert AuditEvent.APPLIED in events


@pytest.mark.asyncio
async def test_an_action_with_no_preview_is_not_approvable(tmp_path: Path) -> None:
    """A proposal whose preview failed is refused, not returned unpreviewed."""
    runner = host_runner(
        **{
            "systemctl --system show": CommandResult(
                argv=[], outcome=Outcome.FAILED, stderr="boom", returncode=1
            )
        }
    )
    core, executor, log = engine(tmp_path, runner=runner)

    with pytest.raises(HostdRefusal) as refused:
        await core.propose(ActionKind.UNIT_STOP, {"unit": "nginx"}, _requester())

    assert refused.value.code is ErrorCode.REFUSED
    assert "not approvable without" in refused.value.detail
    assert executor.calls == []
    assert log.tail(5)[-1].event is AuditEvent.REFUSED


@pytest.mark.asyncio
async def test_host_action_approval_is_scoped_and_single_use(tmp_path: Path) -> None:
    """An approval applies once, to one action, against the system it previewed."""
    core, executor, _log = engine(tmp_path)
    view = await core.propose(ActionKind.UNIT_STOP, {"unit": "nginx"}, _requester())
    await core.apply(view.id, on_output=_ignore, approved_by="operator")

    # Replay: the same approved id a second time.
    with pytest.raises(HostdRefusal) as replay:
        await core.apply(view.id, on_output=_ignore, approved_by="operator")
    assert replay.value.code is ErrorCode.ALREADY_USED
    assert len(executor.calls) == 1

    # Scope: approving one action approves nothing else. A second proposal is a
    # second decision, with its own id.
    other = await core.propose(ActionKind.UNIT_START, {"unit": "nginx"}, _requester())
    assert other.id != view.id
    assert other.state is ProposalState.PENDING


@pytest.mark.asyncio
async def test_approval_after_the_system_moved_is_refused(tmp_path: Path) -> None:
    """The preview described a world that no longer exists, so the answer is no."""
    runner = host_runner()
    core, executor, log = engine(tmp_path, runner=runner)
    view = await core.propose(ActionKind.UNIT_STOP, {"unit": "nginx"}, _requester())

    # Someone stopped it between the preview and the approval.
    runner.results["systemctl --system show"] = UNIT_SHOW_INACTIVE

    with pytest.raises(HostdRefusal) as drifted:
        await core.apply(view.id, on_output=_ignore, approved_by="operator")

    assert drifted.value.code is ErrorCode.DRIFTED
    assert "propose it again" in drifted.value.detail.lower()
    assert executor.calls == []
    assert core.proposal(view.id) is not None
    assert core.proposal(view.id).state is ProposalState.DRIFTED  # type: ignore[union-attr]
    assert any(record.outcome == "drifted" for record in log.tail(10))


@pytest.mark.asyncio
async def test_a_stale_proposal_cannot_be_approved(tmp_path: Path) -> None:
    """The approval window closes, and a closed window is a refusal not a run."""
    moment = [1000.0]
    core, executor, _log = engine(tmp_path, ttl=60.0, clock=lambda: moment[0])
    view = await core.propose(ActionKind.UNIT_START, {"unit": "nginx"}, _requester())

    moment[0] += 3600.0

    with pytest.raises(HostdRefusal) as stale:
        await core.apply(view.id, on_output=_ignore, approved_by="operator")
    assert stale.value.code is ErrorCode.EXPIRED
    assert executor.calls == []


@pytest.mark.asyncio
async def test_concurrent_approvals_of_one_proposal_run_it_once(
    tmp_path: Path,
) -> None:
    """Two operators (or two tabs) racing produce one execution and one refusal."""
    executor = FakeExecutor()
    core, _executor, _log = engine(tmp_path, executor=executor)
    view = await core.propose(ActionKind.UNIT_RESTART, {"unit": "nginx"}, _requester())

    outcomes = await asyncio.gather(
        core.apply(view.id, on_output=_ignore, approved_by="one"),
        core.apply(view.id, on_output=_ignore, approved_by="two"),
        return_exceptions=True,
    )

    applied = [item for item in outcomes if not isinstance(item, BaseException)]
    refused = [item for item in outcomes if isinstance(item, HostdRefusal)]
    assert len(applied) == 1
    assert len(refused) == 1
    assert refused[0].code is ErrorCode.ALREADY_USED
    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_host_action_cancellation_is_recorded(tmp_path: Path) -> None:
    """Cancelling mid-apply leaves a recorded outcome, not an unknown state."""
    executor = FakeExecutor(output=[("stdout", "stopping...\n")], hang=True)
    core, _executor, log = engine(tmp_path, executor=executor)
    view = await core.propose(ActionKind.UNIT_RESTART, {"unit": "nginx"}, _requester())

    task = asyncio.ensure_future(
        core.apply(view.id, on_output=_ignore, approved_by="operator")
    )
    await asyncio.wait_for(executor.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert core.proposal(view.id).state is ProposalState.CANCELLED  # type: ignore[union-attr]
    cancelled = [r for r in log.tail(20) if r.event is AuditEvent.CANCELLED]
    assert len(cancelled) == 1
    # The record says what is and is not known afterwards, rather than implying
    # the action did not happen.
    assert "already done stands" in cancelled[0].detail
    assert cancelled[0].action_id == view.id


@pytest.mark.asyncio
async def test_a_failed_apply_is_recorded_as_failed(tmp_path: Path) -> None:
    executor = FakeExecutor(
        result=CommandResult(
            argv=[], outcome=Outcome.FAILED, returncode=5, stderr="unit not found"
        )
    )
    core, _executor, log = engine(tmp_path, executor=executor)
    view = await core.propose(ActionKind.UNIT_START, {"unit": "nginx"}, _requester())

    result = await core.apply(view.id, on_output=_ignore, approved_by="operator")

    assert not result.ok
    assert result.returncode == 5
    assert log.tail(1)[0].event is AuditEvent.FAILED
    assert core.proposal(view.id).state is ProposalState.FAILED  # type: ignore[union-attr]


# --- audit ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_host_actions_are_audited(tmp_path: Path) -> None:
    """Every point in an action's life produces a record with who and what."""
    core, _executor, log = engine(tmp_path)
    requester = _requester(actor="agent", agent="ops-1", run="run-9")

    denied = await core.propose(ActionKind.UNIT_STOP, {"unit": "nginx"}, requester)
    core.deny(denied.id, operator="alex", reason="not now")

    applied = await core.propose(ActionKind.UNIT_RESTART, {"unit": "nginx"}, requester)
    await core.apply(applied.id, on_output=_ignore, approved_by="alex")

    with pytest.raises(HostdRefusal):
        await core.propose(ActionKind.UNIT_RESTART, {"unit": "sshd"}, requester)

    records = log.tail(50)
    by_event = {record.event: record for record in records}
    assert AuditEvent.REQUESTED in by_event
    assert AuditEvent.DENIED in by_event
    assert AuditEvent.APPROVED in by_event
    assert AuditEvent.APPLIED in by_event
    assert AuditEvent.REFUSED in by_event

    requested = next(r for r in records if r.event is AuditEvent.REQUESTED)
    assert requested.requester.agent == "ops-1"
    assert requested.requester.run == "run-9"
    assert requested.steps[0].argv == ["systemctl", "stop", "--", "nginx.service"]
    assert requested.reversal  # what the inverse is, recorded at request time

    approved = by_event[AuditEvent.APPROVED]
    assert approved.requester.actor == "alex"

    applied_record = by_event[AuditEvent.APPLIED]
    assert applied_record.returncode == 0
    assert applied_record.restore_point  # the state it can be put back to


@pytest.mark.asyncio
async def test_the_audit_redacts_secret_shaped_values(tmp_path: Path) -> None:
    """The record of a privileged action must not become the leak."""
    core, _executor, log = engine(tmp_path)
    core.record_refusal("a caller sent secret=s3cr3t-value on the socket")

    line = (tmp_path / "audit.jsonl").read_text()
    assert "s3cr3t-value" not in line
    assert "[redacted]" in line


# --- reversal ------------------------------------------------------------


@pytest.mark.asyncio
async def test_reversal_is_recorded_or_declared_impossible(tmp_path: Path) -> None:
    """A reversible action carries its inverse; a one-way action says so."""
    core, _executor, _log = engine(tmp_path)

    stop = await core.propose(ActionKind.UNIT_STOP, {"unit": "nginx"}, _requester())
    assert stop.reversal.possible
    assert stop.reversal.kind is ActionKind.UNIT_START
    assert stop.reversal.args == {"unit": "nginx.service"}

    # And the inverse is a real action this helper will build.
    inverse = await core.propose(
        stop.reversal.kind, dict(stop.reversal.args), _requester()
    )
    assert inverse.steps[0].argv == ["systemctl", "start", "--", "nginx.service"]

    restart = await core.propose(
        ActionKind.UNIT_RESTART, {"unit": "nginx"}, _requester()
    )
    assert not restart.reversal.possible
    assert "cannot be undone" in restart.reversal.summary

    gc = await core.propose(ActionKind.GC_STORE, {}, _requester())
    assert not gc.reversal.possible
    assert "ONE-WAY" in gc.reversal.summary


# --- refusals that have no code path ------------------------------------


def test_a_unit_name_that_would_become_an_option_is_refused() -> None:
    """`-Hsomeone@host` made systemctl open an SSH connection in 20260729-125024."""
    with pytest.raises(ActionRefused) as refused:
        normalise_unit("-Hsomeone@host")
    assert "would be read as an option" in str(refused.value)

    with pytest.raises(ActionRefused):
        normalise_unit("nginx.service; rm -rf /")
    with pytest.raises(ActionRefused):
        normalise_unit("../../etc/passwd")


def test_units_that_would_cut_the_operator_off_are_refused() -> None:
    for unit in ("sshd", "sshd.service", "NetworkManager", "dbus", "systemd-logind"):
        with pytest.raises(ActionRefused) as refused:
            normalise_unit(unit)
        assert "deny-list" in str(refused.value)


def test_targets_slices_and_scopes_have_no_code_path_at_all() -> None:
    """The deny-list is a stem list, so the dangerous names walked around it.

    `emergency.target` drops the box to single-user and kills sshd - the exact
    outcome the `sshd` entry exists to prevent, through a name no stem list
    catches. Refusing the whole unit TYPE is a boundary; enumerating the
    dangerous names inside it is a game of catch-up (review round 1, R1.5).
    """
    for unit in (
        "emergency.target",
        "rescue.target",
        "multi-user.target",
        "graphical.target",
        "network.target",
        "user.slice",
        "system.slice",
        "init.scope",
        "session-1.scope",
    ):
        with pytest.raises(ActionRefused) as refused:
            normalise_unit(unit)
        assert "not targets, slices or scopes" in str(refused.value)


def test_a_templated_instance_is_refused_like_its_template() -> None:
    """`user@1000.service` kills the operator's session - and scufris with it.

    scufris runs as a USER service on this host, so ending the session manager
    ends the approval path. The stem check alone missed it: the stem is
    `user@1000`, which is not in any list.
    """
    for unit in (
        "user@1000.service",
        "user@0.service",
        "user-runtime-dir@1000.service",
        "getty@tty1.service",
    ):
        with pytest.raises(ActionRefused) as refused:
            normalise_unit(unit)
        assert "deny-list" in str(refused.value)


def test_the_units_an_operator_actually_means_still_work() -> None:
    """The paired guard: the refusals above must not have refused everything.

    Without this, tightening the list until nothing is allowed would look
    identical to tightening it correctly.
    """
    assert normalise_unit("nginx") == "nginx.service"
    assert normalise_unit("photo-gallery.service") == "photo-gallery.service"
    assert normalise_unit("podman.socket") == "podman.socket"
    assert normalise_unit("nixos-upgrade.timer") == "nixos-upgrade.timer"
    assert normalise_unit("borgbackup-job-home@daily.service") == (
        "borgbackup-job-home@daily.service"
    )


def test_the_helper_refuses_to_act_on_scufris_or_itself() -> None:
    for unit in ("scufris", "scufris.service", "scufris-hostd.service"):
        with pytest.raises(ActionRefused) as refused:
            normalise_unit(unit)
        assert "may not control" in str(refused.value)


def test_a_bare_name_becomes_a_service_and_an_unknown_type_is_refused() -> None:
    assert normalise_unit("nginx") == "nginx.service"
    assert normalise_unit("nginx.socket") == "nginx.socket"
    with pytest.raises(ActionRefused) as refused:
        normalise_unit("nginx.wat")
    assert "unknown unit type" in str(refused.value)


def test_there_is_no_shell_verb() -> None:
    """The taxonomy is the verb set; a shell verb would make it decorative."""
    names = {kind.value for kind in ActionKind}
    assert not {
        name for name in names if "shell" in name or "exec" in name or "run" in name
    }
    assert names == {
        "unit_start",
        "unit_stop",
        "unit_restart",
        "unit_reload",
        "gc_older_than",
        "gc_store",
        "activate",
        "rollback",
    }


# --- the R2 generation floor --------------------------------------------


def test_garbage_collection_never_eats_the_rollback_target() -> None:
    """The floor has to be in the ARGV, not just in the display list.

    Asserting `generations_removed` alone is what let the original bug through:
    the preview said 190 was kept while the emitted
    `nix-collect-garbage --delete-older-than 1d` would have deleted it, because
    that flag keeps only the CURRENT generation and is otherwise purely
    age-based. So this test reads the COMMAND (review round 1, R1.4).
    """
    runner = host_runner()
    plan = build_plan(ActionKind.GC_OLDER_THAN, {"days": 1}, runner=runner, now=NOW)

    assert plan.steps[0].argv == [
        "nix-env",
        "--profile",
        "/nix/var/nix/profiles/system",
        "--delete-generations",
        "180",
        "179",
    ]
    # The two most recent are absent from the COMMAND, not merely from a list
    # rendered next to it.
    assert "191" not in plan.steps[0].argv
    assert "190" not in plan.steps[0].argv
    assert "--delete-older-than" not in plan.steps[0].argv
    assert plan.generations_removed == [180, 179]


def test_a_collection_with_nothing_old_enough_is_refused() -> None:
    """An approval must never be asked for a command that would do nothing."""
    runner = host_runner()
    with pytest.raises(ActionRefused) as refused:
        build_plan(ActionKind.GC_OLDER_THAN, {"days": 3650}, runner=runner, now=NOW)
    assert "nothing to delete" in str(refused.value)


def test_the_floor_holds_on_a_box_nobody_has_rebuilt_in_a_year() -> None:
    """The case the flag gets wrong: every generation is older than the cutoff."""
    runner = host_runner()
    plan = build_plan(
        ActionKind.GC_OLDER_THAN,
        {"days": 1},
        runner=runner,
        now=datetime(2030, 1, 1),
    )
    assert "191" not in plan.steps[0].argv
    assert "190" not in plan.steps[0].argv
    assert plan.generations_removed == [180, 179]


def test_generations_with_an_unreadable_date_are_kept() -> None:
    """ "We could not tell how old it is" must never resolve to "delete it"."""
    from scufris.host.storage import Generation

    generations = [
        Generation(number=n, date="not a date" if n < 100 else "2020-01-01 00:00:00")
        for n in (200, 199, 150, 99)
    ]
    removed = generations_older_than(generations, 1, now=NOW)
    assert [g.number for g in removed] == [150]


def test_the_floor_is_two_generations() -> None:
    assert PROTECTED_GENERATIONS == 2


def test_a_collection_is_refused_when_the_generation_list_is_unreadable() -> None:
    """Without the list the floor cannot be checked, so the answer is no."""
    runner = host_runner(
        **{
            "nixos-rebuild list-generations": CommandResult(
                argv=[], outcome=Outcome.DENIED, stderr="permission denied"
            )
        }
    )
    with pytest.raises(ActionRefused) as refused:
        build_plan(ActionKind.GC_OLDER_THAN, {"days": 30}, runner=runner, now=NOW)
    assert "floor cannot be checked" in str(refused.value)


def test_gc_is_never_the_bare_delete_everything_form() -> None:
    runner = host_runner()
    plan = build_plan(ActionKind.GC_OLDER_THAN, {"days": 30}, runner=runner, now=NOW)
    assert "-d" not in plan.steps[0].argv
    assert "nix-collect-garbage" not in plan.steps[0].argv
    assert plan.steps[0].argv[:4] == [
        "nix-env",
        "--profile",
        "/nix/var/nix/profiles/system",
        "--delete-generations",
    ]
    # Every remaining argument is a generation number, so none can be an option.
    assert all(part.isdigit() for part in plan.steps[0].argv[4:])


def test_garbage_collection_days_are_bounded() -> None:
    runner = host_runner()
    for days in (0, -1, 100000):
        with pytest.raises(ActionRefused):
            build_plan(ActionKind.GC_OLDER_THAN, {"days": days}, runner=runner, now=NOW)


# --- previews say what they are -----------------------------------------


@pytest.mark.asyncio
async def test_a_service_preview_is_labelled_state_not_simulation(
    tmp_path: Path,
) -> None:
    core, _executor, _log = engine(tmp_path)
    view = await core.propose(ActionKind.UNIT_STOP, {"unit": "nginx"}, _requester())

    assert view.preview.kind is PreviewKind.STATE
    assert "cannot simulate" in view.preview.label
    body = "\n".join(view.preview.lines)
    assert "now:  active (running)" in body
    assert "after: inactive" in body
    assert "multi-user.target" in body  # the blast radius


@pytest.mark.asyncio
async def test_a_gc_preview_reports_freed_space_not_a_path_count(
    tmp_path: Path,
) -> None:
    """The count is a count and the size is a size; neither wears the other's name."""
    core, _executor, _log = engine(tmp_path)
    view = await core.propose(ActionKind.GC_STORE, {}, _requester())

    body = "\n".join(view.preview.lines)
    assert "store paths already unreachable: 2" in body
    # 1024 + 2048 narSize, summed per path - not a sum of overlapping closures.
    assert "3.0 KiB" in body


@pytest.mark.asyncio
async def test_an_empty_gc_preview_says_nothing_is_dead(tmp_path: Path) -> None:
    """Zero is an answer. It must not render like a preview that broke."""
    core, _executor, _log = engine(
        tmp_path, runner=host_runner(**{"nix-store --gc --print-dead": ok_result("")})
    )
    view = await core.propose(ActionKind.GC_STORE, {}, _requester())

    assert view.preview.ok
    assert "would delete nothing" in "\n".join(view.preview.lines)


@pytest.mark.asyncio
async def test_a_gc_preview_lists_the_generations_and_keeps_the_floor(
    tmp_path: Path,
) -> None:
    core, _executor, _log = engine(tmp_path)
    view = await core.propose(ActionKind.GC_OLDER_THAN, {"days": 1}, _requester())

    body = "\n".join(view.preview.lines)
    assert "generations this deletes (2)" in body
    assert "191" in body and "190" in body  # listed under "kept"
    # The preview names the same generations the COMMAND does - it is derived
    # from the argv rather than computed a second time alongside it, which is
    # how the two came to disagree in the first place.
    assert "180" in view.steps[0].argv and "179" in view.steps[0].argv
    # And it does not claim to free space, because this action does not.
    assert "frees no disk space by itself" in body
    assert "space freed" not in body


def _requester(actor: str = "operator", agent: str = "", run: str = ""):
    from scufris.hostd import Requester

    return Requester(actor=actor, agent=agent, run=run)


def _ignore(stream: str, text: str) -> None:
    return None


@pytest.mark.asyncio
async def test_a_requester_cannot_hold_more_than_the_pending_cap(
    tmp_path: Path,
) -> None:
    """Proposing is free of consequence but not free of COST.

    The R2 previews walk the store under the global nix GC lock, so a caller
    holding only the machine token could loop `propose` and keep that lock
    contended as root while never approving anything (review round 1, R1.10).
    """
    from scufris.hostd.engine import MAX_PENDING_PER_REQUESTER

    core, executor, _ = engine(tmp_path)
    who = _requester(actor="agent", agent="orchestrator")

    for _ in range(MAX_PENDING_PER_REQUESTER):
        await core.propose(ActionKind.UNIT_RESTART, {"unit": "nginx"}, who)

    with pytest.raises(HostdRefusal) as refused:
        await core.propose(ActionKind.UNIT_RESTART, {"unit": "nginx"}, who)
    assert refused.value.code is ErrorCode.REFUSED
    assert "waiting for a decision" in str(refused.value)
    assert executor.calls == []


@pytest.mark.asyncio
async def test_varying_the_agent_name_does_not_raise_the_proposal_cap(
    tmp_path: Path,
) -> None:
    """The cap keys on the CREDENTIAL, not on a name the caller picks.

    The first version keyed on `requester.agent or requester.actor`, and `agent`
    is a body field: a machine caller varying it per request held twenty pending
    proposals against a cap of five (review round 2, R2.2). It is R1.6's lesson
    again - the caller must not control the identity the server decides on.
    """
    from scufris.hostd.engine import MAX_PENDING_PER_REQUESTER

    core, _, _ = engine(tmp_path)

    accepted = 0
    with pytest.raises(HostdRefusal):
        for i in range(MAX_PENDING_PER_REQUESTER * 4):
            await core.propose(
                ActionKind.UNIT_RESTART,
                {"unit": "nginx"},
                _requester(actor="agent", agent=f"orchestrator-{i}"),
            )
            accepted += 1
    assert accepted == MAX_PENDING_PER_REQUESTER


@pytest.mark.asyncio
async def test_the_cap_is_per_requester_not_global(tmp_path: Path) -> None:
    """The paired guard: capping the agent must not lock the operator out."""
    from scufris.hostd.engine import MAX_PENDING_PER_REQUESTER

    core, _, _ = engine(tmp_path)
    for _ in range(MAX_PENDING_PER_REQUESTER):
        await core.propose(
            ActionKind.UNIT_RESTART, {"unit": "nginx"}, _requester(actor="agent")
        )

    view = await core.propose(
        ActionKind.UNIT_RESTART, {"unit": "nginx"}, _requester(actor="operator:abc123")
    )
    assert view.state is ProposalState.PENDING


@pytest.mark.asyncio
async def test_pending_lists_only_what_is_still_decidable(tmp_path: Path) -> None:
    """`list_pending` is how the app gets a truthful queue back after a restart, so
    it must never report a proposal that an apply would refuse.

    It expires the stale ones FIRST and then lists, so the answer and the decision
    read the same state: a proposal past its window is absent from the list AND
    refused by apply, rather than being offered and then rejected.
    """
    core, executor, log = engine(tmp_path, ttl=60.0)
    live = await core.propose(ActionKind.UNIT_RESTART, {"unit": "nginx"}, _requester())
    gone = await core.propose(ActionKind.GC_STORE, {}, _requester())

    listed = core.pending()
    assert [p.id for p in listed.proposals] == [live.id, gone.id]
    # The whole proposal comes back, not a stub: this is what the app rebuilds its
    # queue from, so the preview and the commands have to survive.
    assert listed.proposals[0].preview.lines
    assert listed.proposals[0].steps[0].argv == [
        "systemctl",
        "restart",
        "--",
        "nginx.service",
    ]
    assert listed.proposals[0].requester.actor == _requester().actor

    # A decided one drops out, whichever way it was decided: applied is not
    # pending, and neither is denied.
    await core.apply(gone.id, on_output=_ignore, approved_by="operator")
    assert executor.calls, "the applied proposal never ran"
    assert [p.id for p in core.pending().proposals] == [live.id]
    core.deny(live.id, operator="operator", reason="not now")
    assert core.pending().proposals == []
    events = [record.event for record in log.tail(20)]
    assert AuditEvent.APPLIED in events and AuditEvent.DENIED in events


@pytest.mark.asyncio
async def test_pending_expires_a_stale_proposal_before_answering(
    tmp_path: Path,
) -> None:
    """The sweep is part of the read, not a side effect someone else has to trigger."""
    clock = {"now": 1000.0}
    core, _executor, log = engine(tmp_path, ttl=10.0, clock=lambda: clock["now"])
    view = await core.propose(ActionKind.UNIT_RESTART, {"unit": "nginx"}, _requester())
    assert [p.id for p in core.pending().proposals] == [view.id]

    clock["now"] += 11.0
    assert core.pending().proposals == []
    assert AuditEvent.EXPIRED in [record.event for record in log.tail(20)]
    with pytest.raises(HostdRefusal) as expired:
        await core.apply(view.id, on_output=_ignore, approved_by="operator")
    assert expired.value.code is ErrorCode.EXPIRED
