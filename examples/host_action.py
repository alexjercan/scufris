#!/usr/bin/env python
"""Drive one host action through the whole contract and print what an operator sees.

    propose -> preview -> approve -> apply -> audit -> revert

    python examples/host_action.py              # the reversible case (a unit stop)
    python examples/host_action.py --one-way    # the case with NO undo (a store gc)
    python examples/host_action.py --deny       # the operator says no
    python examples/host_action.py --cancel     # stopped mid-apply

Nothing here touches the real machine. The engine takes a ``FakeRunner``
replaying this host's real command output and a ``FakeExecutor`` that records
what it was asked to run instead of running it, so the whole path - including
the audit log and the cancellation - is exercisable without root, without a
socket, and without a NixOS box.

That is also what it is FOR: the approval text is the operator-facing half of
this feature, so this is where the wording gets read before anyone trusts it. For
the AGENT's round trip - proposing with the credential it really holds, being left
BLOCKED, and being resumed with the decision - see `examples/host_agent.py`; the
dashboard and Telegram approval surfaces are 20260730-104520 and 20260730-104524.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

# Run from a checkout without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scufris.db import open_state_database  # noqa: E402
from scufris_host import FakeRunner, ok_result  # noqa: E402
from scufris_hostctl import (  # noqa: E402
    HostActionStore,
    confirmation_for,  # noqa: E402
    render_action,
)
from scufris_hostd import (  # noqa: E402
    ActionKind,
    AuditLog,
    FakeExecutor,
    HostdEngine,
    HostdRefusal,
    Requester,
)

# What this host's commands really print, captured from the box in
# tasks/20260729-125020/SPIKE.md.
RUNNER = FakeRunner(
    results={
        "systemctl --system show": ok_result(
            "Id=nginx.service\n"
            "Description=Nginx Web Server\n"
            "LoadState=loaded\n"
            "ActiveState=active\n"
            "SubState=running\n"
            "UnitFileState=enabled\n"
            "MainPID=4242\n"
        ),
        "systemctl list-dependencies": ok_result(
            "nginx.service\nmulti-user.target\nphoto-gallery.service\n"
        ),
        "nixos-rebuild list-generations": ok_result(
            json.dumps(
                [
                    {"generation": 191, "date": "2026-07-29 10:00:00", "current": True},
                    {"generation": 190, "date": "2026-07-28 10:00:00"},
                    {"generation": 180, "date": "2026-01-01 10:00:00"},
                ]
            )
        ),
        "nix-store --gc --print-dead": ok_result(
            "\n".join(f"/nix/store/{index:040x}-thing" for index in range(2000))
        ),
        "nix path-info": ok_result(
            json.dumps(
                {
                    f"/nix/store/{index:040x}-thing": {"narSize": 3_500_000}
                    for index in range(2000)
                }
            )
        ),
        "nix-collect-garbage --dry-run": ok_result(
            "7642 store paths would be deleted\n"
        ),
    }
)


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--one-way",
        action="store_true",
        help="propose a store collection - the class that cannot be undone",
    )
    parser.add_argument(
        "--deny", action="store_true", help="have the operator refuse the proposal"
    )
    parser.add_argument(
        "--cancel", action="store_true", help="stop the apply while it is running"
    )
    args = parser.parse_args()

    directory = Path(tempfile.mkdtemp(prefix="host-action-example-"))
    audit = AuditLog(directory / "audit.jsonl")
    executor = FakeExecutor(output=[("stdout", "working...\n")], hang=args.cancel)
    engine = HostdEngine(audit, runner=RUNNER, executor=executor)
    # The store is a table now, so the example opens the same state database the
    # app does - migrations and all - under its own throwaway directory.
    store = HostActionStore(await asyncio.to_thread(open_state_database, directory))

    kind = ActionKind.GC_STORE if args.one_way else ActionKind.UNIT_STOP
    action_args: dict[str, object] = {} if args.one_way else {"unit": "nginx"}

    # 1. PROPOSE. An agent asks; nothing happens yet.
    banner("1. the agent proposes - and nothing has happened")
    try:
        proposal = await engine.propose(
            kind, action_args, Requester(actor="agent", agent="ops-1", run="run-9")
        )
    except HostdRefusal as refusal:
        print(f"refused ({refusal.code}): {refusal.detail}")
        return 1
    # Every store call here is OFFLOADED: the store opens a transaction, and a
    # transaction cannot be held on a thread with a running event loop. The app
    # follows the same rule everywhere it touches a store.
    record = await asyncio.to_thread(store.put, proposal)
    print(render_action(record))
    print(f"\n(the executor has run {len(executor.calls)} commands)")

    # 2. The operator decides.
    if args.deny:
        banner("2. the operator says no")
        engine.deny(proposal.id, operator="alex", reason="not during the week")
        await asyncio.to_thread(
            store.deny, proposal.id, operator="alex", reason="not during the week"
        )
        print(render_action(await asyncio.to_thread(store.get, proposal.id)))
        print(f"\n(the executor has run {len(executor.calls)} commands)")
        return _print_audit(audit)

    # What the operator must DO to approve this, computed by the same function both
    # approval surfaces render from - so an action that cannot be undone cannot be
    # approved through the ordinary confirmation on any of them.
    confirmation = confirmation_for(proposal)
    banner("2. the operator approves - THIS is what runs it")
    print(f"risk:    {confirmation.risk_label}")
    print(f"undo:    {confirmation.undo}")
    if confirmation.one_way:
        print(
            f"confirm: this one is ONE-WAY, so approving it requires typing "
            f"{confirmation.acknowledge!r} - the ordinary confirmation is refused.\n"
        )
    else:
        print("confirm: ordinary (the undo above is what makes that enough)\n")
    await asyncio.to_thread(store.approve, proposal.id, operator="alex")

    def show(stream: str, text: str) -> None:
        print(f"  [{stream}] {text}", end="")

    task = asyncio.ensure_future(
        engine.apply(proposal.id, on_output=show, approved_by="alex")
    )
    if args.cancel:
        await asyncio.wait_for(executor.started.wait(), timeout=5)
        banner("3. the operator stops it mid-apply")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print("cancelled. What it had already done still stands - and is recorded.")
        await asyncio.to_thread(store.finish, proposal.id, error="cancelled mid-apply")
    else:
        result = await task
        await asyncio.to_thread(store.finish, proposal.id, result=result)
        banner("3. the result")
        print(render_action(await asyncio.to_thread(store.get, proposal.id)))

    # 4. REVERT - itself a proposal, with its own preview and its own approval.
    banner("4. undoing it")
    reversal = proposal.reversal
    if not reversal.possible or reversal.kind is None:
        print(f"NO UNDO: {reversal.summary}")
    else:
        inverse = await engine.propose(
            reversal.kind, dict(reversal.args), Requester(actor="alex")
        )
        print("The undo is ITSELF a proposal - it needs its own approval:\n")
        print(render_action(await asyncio.to_thread(store.put, inverse)))

    return _print_audit(audit)


def _print_audit(audit: AuditLog) -> int:
    banner("the record the helper wrote (root-owned and append-only in production)")
    for entry in audit.tail(20):
        print(
            f"{entry.at}  {entry.event:<10} {entry.action_id[:8] or '-':<8} "
            f"{' '.join((entry.steps[0].argv if entry.steps else [])) or entry.detail}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
