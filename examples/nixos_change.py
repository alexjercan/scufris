#!/usr/bin/env python
"""Drive a NixOS configuration change end to end and print what an operator sees.

    resolve -> build -> preview -> approve -> switch -> roll back

    python examples/nixos_change.py             # a change that alters the closure
    python examples/nixos_change.py --no-change # the built system is what is running
    python examples/nixos_change.py --fail      # the configuration does not build
    python examples/nixos_change.py --busy      # another switch is already running

It builds a REAL temporary git repository (so the resolve half is real git, on a
real commit, on a real branch) and fakes exactly two things: the `nix build`,
which would take minutes and needs a nixpkgs, and the privileged commands, which
would need root. Everything between them - the closure-diff preview, the
approval, the two-step activation, the audit, the rollback - is the shipped code.

That is also what it is FOR. The approval text is the operator-facing half of
this feature and the dashboard surface (20260729-125040) is not built yet, so
this is where the wording gets read before anyone trusts it with a machine.

What the example CANNOT show is a real activation: that needs root and a real
system profile, and it is proved in `nix build .#scufris-hostd-vm-test`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Run from a checkout without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scufris.db import open_state_database  # noqa: E402
from scufris.host_actions import HostActionStore, render_action  # noqa: E402
from scufris.hostconfig import (  # noqa: E402
    ConfigChange,
    ConfigChangeBuilder,
    ConfigChangeStore,
    render_change,
)
from scufris_host import (  # noqa: E402
    NIX_FEATURES,
    CommandResult,
    FakeRunner,
    Outcome,
    ok_result,
    run_command,
)
from scufris_hostd import (  # noqa: E402
    ActionKind,
    AuditLog,
    FakeExecutor,
    FakeFiles,
    HostdEngine,
    HostdRefusal,
    Requester,
)

# The two systems in play. Real store-path shapes: a 32-character nix base-32
# hash, whose alphabet has no e, o, t or u.
# A new-CLI nix invocation carries its experimental features explicitly, so the
# fake runner's keys do too.
NIX = " ".join(["nix", *NIX_FEATURES])

RUNNING = "/nix/store/bnfi69bsjhaj4jgp42jk9ys6y80pb9qh-nixos-system-nixos-26.11"
BUILT = "/nix/store/c0z2q4wl5m7dnpx9rsv0abcdfghijklm-nixos-system-nixos-26.11"

# What `nix store diff-closures` really prints, colour codes and glyphs included
# (measured on this host: NO_COLOR does not silence them, so the preview strips
# them itself).
CLOSURE_DIFF = ok_result(
    "acl: 2.3.2 \u2192 2.4.0, \x1b[31;1m30.4 KiB\x1b[0m\n"
    "ripgrep: \u2205 \u2192 14.1.1, \x1b[31;1m4.1 MiB\x1b[0m\n"
    "linux: 6.18.40 \u2192 6.18.41, \x1b[31;1m2.1 MiB\x1b[0m\n"
)

GENERATIONS = ok_result(
    json.dumps(
        [
            {
                "generation": 191,
                "date": "2026-07-29 10:00:00",
                "nixosVersion": "26.11",
                "current": True,
            },
            {"generation": 190, "date": "2026-07-28 10:00:00", "nixosVersion": "26.11"},
        ]
    )
)

GENERATIONS_AFTER = ok_result(
    json.dumps(
        [
            {
                "generation": 192,
                "date": "2026-07-29 12:30:00",
                "nixosVersion": "26.11",
                "current": True,
            },
            {"generation": 191, "date": "2026-07-29 10:00:00", "nixosVersion": "26.11"},
            {"generation": 190, "date": "2026-07-28 10:00:00", "nixosVersion": "26.11"},
        ]
    )
)

INACTIVE = CommandResult(
    argv=[], outcome=Outcome.FAILED, stdout="inactive\n", returncode=3
)


def host(*, no_change: bool = False, busy: bool = False) -> FakeRunner:
    """The machine, as the privileged helper reads it."""
    return FakeRunner(
        results={
            "nixos-rebuild list-generations": GENERATIONS,
            f"{NIX} path-info": ok_result(""),
            f"{NIX} store diff-closures": ok_result("") if no_change else CLOSURE_DIFF,
            "systemctl is-active": ok_result("active\n") if busy else INACTIVE,
        }
    )


def host_files() -> FakeFiles:
    return FakeFiles(
        files={f"{path}/nixos-version" for path in (RUNNING, BUILT)},
        executables={
            f"{path}/bin/switch-to-configuration" for path in (RUNNING, BUILT)
        },
        links={
            "/run/current-system": RUNNING,
            "/nix/var/nix/profiles/system-191-link": RUNNING,
        },
    )


class BuildExecutor(FakeExecutor):
    """A `nix build` that prints a store path, and privileged commands that pass."""

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail

    async def __call__(  # type: ignore[override]
        self, argv: list[str], *, timeout: float, on_output: Any
    ) -> CommandResult:
        self.calls.append(list(argv))
        if argv[0] == "nix" and "build" in argv:
            on_output("stderr", "these 412 derivations will be built:\n")
            on_output("stderr", "  /nix/store/....-nixos-system-nixos-26.11.drv\n")
            if self.fail:
                on_output(
                    "stderr",
                    "error: attribute 'ripgrp' missing\n"
                    "       at /nix/store/...-source/hosts/nixos/default.nix:204:5\n",
                )
                return CommandResult(
                    argv=argv, outcome=Outcome.FAILED, returncode=1, stderr="error"
                )
            return CommandResult(
                argv=argv, outcome=Outcome.OK, stdout=f"{BUILT}\n", returncode=0
            )
        on_output("stdout", "setting up /etc...\nreloading user units for alex...\n")
        return CommandResult(argv=argv, outcome=Outcome.OK, returncode=0)


def make_repo(root: Path) -> Path:
    """A real git repository with a flake and a committed change on a branch.

    Real on purpose: resolving a ref, reading its subject, noticing that it is
    not merged and noticing uncommitted files are all git behaviour, and faking
    git would only prove the fake works.
    """
    repo = root / "nix.dotfiles"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            env={
                "HOME": str(root),
                # From the environment rather than a guess, so this runs the same
                # way inside a nix shell as outside one.
                "PATH": os.environ.get("PATH", ""),
                "GIT_AUTHOR_NAME": "scufris",
                "GIT_AUTHOR_EMAIL": "scufris@localhost",
                "GIT_COMMITTER_NAME": "scufris",
                "GIT_COMMITTER_EMAIL": "scufris@localhost",
            },
        )

    git("init", "-q", "-b", "master")
    (repo / "flake.nix").write_text("{ outputs = _: {}; }\n")
    (repo / "packages.nix").write_text("[ git vim ]\n")
    git("add", ".")
    git("commit", "-qm", "initial")
    git("checkout", "-qb", "config/add-ripgrep")
    (repo / "packages.nix").write_text("[ git ripgrep vim ]\n")
    git("commit", "-qam", "feat(packages): add ripgrep to the system profile")
    return repo


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-change",
        action="store_true",
        help="the built system is identical to the running one (the measured trap)",
    )
    parser.add_argument(
        "--fail", action="store_true", help="the configuration does not build"
    )
    parser.add_argument(
        "--busy",
        action="store_true",
        help="another switch-to-configuration is already running",
    )
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="nixos-change-example-"))
    repo = make_repo(root)
    audit = AuditLog(root / "audit.jsonl")
    runner = host(no_change=args.no_change, busy=args.busy)
    files = host_files()
    executor = BuildExecutor(fail=args.fail)
    engine = HostdEngine(audit, runner=runner, executor=executor, files=files)
    # The stores are tables now, so the example opens the same state database the
    # app does - migrations and all - under its own throwaway directory.
    db = await asyncio.to_thread(open_state_database, root)
    actions = HostActionStore(db)
    changes = ConfigChangeStore(db)
    # The build's own seams: git for real, the build faked.
    builder = ConfigChangeBuilder(runner=run_command, executor=executor)

    # 1. RESOLVE. The agent has already edited and committed on a branch, the way
    #    it would for any project. This is where scufris picks the story up.
    banner("1. resolve the ref an agent committed on - a project act, not a host one")
    main_repo, resolved = builder.resolve(repo, "config/add-ripgrep")
    change = await asyncio.to_thread(
        changes.put,
        ConfigChange(
            id="example",
            resolved=resolved,
            attr="nixos",
            agent="ops-1",
            requested_by="agent",
        ),
    )
    print(render_change(change))

    # 2. BUILD. Unprivileged, from the COMMIT, streamed.
    banner("2. build that commit as the operator - not as root")

    async def propose(built: ConfigChange) -> str:
        proposal = await engine.propose(
            ActionKind.ACTIVATE,
            {
                "toplevel": built.toplevel,
                "repo": built.resolved.repo,
                "rev": built.resolved.rev,
            },
            Requester(actor="agent", agent="ops-1", run="run-9"),
        )
        # Offloaded: the store opens a transaction, which cannot be held on a
        # thread with a running event loop. The app follows the same rule.
        await asyncio.to_thread(actions.put, proposal)
        return proposal.id

    async def save(built: ConfigChange) -> None:
        # The builder holds no store, so each transition is written back here -
        # offloaded for the same reason the proposal above is.
        await asyncio.to_thread(changes.put, built)

    try:
        async for event in builder.stream(change, propose, save):
            if event.type == "output":
                print(f"  | {event.text.rstrip()}")
            elif event.type == "error":
                print(f"  ! {event.detail}")
    except HostdRefusal as refused:
        print(f"  ! the helper refused the built configuration: {refused.detail}")

    print()
    print(render_change(change))
    if not change.action_id:
        banner("the change never became approvable, and that is the whole point")
        print(
            "A configuration that does not build produces no proposal, so there is\n"
            "nothing an approval could act on. The log is on the record above."
        )
        return 0

    # 3. PREVIEW. What the operator reads before they decide.
    banner("3. the preview the operator decides on")
    record = await asyncio.to_thread(actions.get, change.action_id)
    print(render_action(record))

    # 4. APPROVE and switch.
    banner("4. approve - the profile moves, then the system switches")
    try:
        result = await engine.apply(
            record.proposal.id,
            on_output=lambda stream, text: print(f"  | {text.rstrip()}"),
            approved_by="operator:1a2b3c4d",
        )
    except HostdRefusal as refused:
        print(f"  ! refused: {refused.detail}")
        banner("nothing moved, which is the answer")
        print(
            "The profile is untouched: an activation that starts while another\n"
            "switch is running leaves a system matching neither configuration."
        )
        return 0
    await asyncio.to_thread(actions.finish, record.proposal.id, result=result)
    print(f"\n  {'succeeded' if result.ok else 'FAILED'}: {result.outcome}")
    print(f"  steps completed: {result.steps_completed}/{result.steps_total}")
    print(f"  undo:            {result.reversal.summary}")

    # The machine has switched, so the faked host moves with it.
    runner.results["nixos-rebuild list-generations"] = GENERATIONS_AFTER
    files.links["/run/current-system"] = BUILT
    files.links["/nix/var/nix/profiles/system-192-link"] = BUILT

    # 5. ROLL BACK. An undo is a proposal of its own.
    banner("5. roll back - which is itself a proposal with its own preview")
    reversal = record.proposal.reversal
    assert reversal.kind is not None
    rollback = await engine.propose(
        reversal.kind, dict(reversal.args), Requester(actor="operator:1a2b3c4d")
    )
    print(render_action(await asyncio.to_thread(actions.put, rollback)))
    rolled = await engine.apply(
        rollback.id,
        on_output=lambda stream, text: print(f"  | {text.rstrip()}"),
        approved_by="operator:1a2b3c4d",
    )
    print(f"\n  rollback {'succeeded' if rolled.ok else 'FAILED'}")

    # 6. THE RECORD.
    banner("6. what the root-written audit log says happened")
    for entry in audit.tail(30):
        command = " ".join(entry.steps[0].argv) if entry.steps else entry.detail
        print(f"  {entry.at}  {entry.event:<10}  {entry.kind or '-':<9}  {command}")

    print(f"\ncommands the helper was asked to run ({len(executor.calls)}):")
    for call in executor.calls:
        print(f"  $ {' '.join(call)}")
    print(f"\nthe repository was never written to: {main_repo}")
    print(f"the audit log is at {audit.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
