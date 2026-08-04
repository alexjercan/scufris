#!/usr/bin/env python
"""The unprivileged client's whole job: hold a root action for an operator.

    python examples/hostctl_approval_flow.py

`scufris_hostd` owns the verbs and runs as root; `scufris_hostctl` is the
client that DRIVES it. This script is that client, end to end:

    1. propose  - ask the helper to preview an action, and queue the proposal
    2. render   - what the operator is deciding against: risk, undo, preview
    3. refuse   - a ONE-WAY action needs a typed acknowledgement, and the
                  SERVICE checks it, so a surface cannot decide to skip it
    4. approve  - claim the decision, then dispatch and watch the apply stream
    5. audit    - the root-owned record, read back over the socket

`examples/hostd_socket_roundtrip.py` covers the same contract from the OTHER
side, speaking the wire by hand to prove the helper's boundary is real. This
one imports nothing of the wire: it holds a database, a decision journal and a
supervisor, and the only thing it knows about the helper is `HostdClient`.

Nothing here needs root, a network, or a NixOS machine. The helper runs
in-process on a temporary unix socket - a real one, because `HostdClient` has
no other transport and an example that faked it would prove nothing - over a
`FakeRunner` replaying canned command output and a `FakeExecutor` scripting the
apply. The database is a temporary file.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

# Run from a checkout without installing it. Four members: the client, the
# helper it drives, the read-only inspection the previews are built from, and
# the shared database and supervisor.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _member in ("hostctl", "hostd", "host", "core"):
    sys.path.insert(0, str(_REPO_ROOT / "packages" / _member / "src"))

from scufris_core import Base, open_database  # noqa: E402
from scufris_host import FakeRunner, ok_result  # noqa: E402
from scufris_hostctl import (  # noqa: E402
    ConfirmationRequired,
    HostActionStore,
    HostApplyOutput,
    HostApplyResult,
    HostApprovalService,
    HostdClient,
    host_supervisor,
    render_action,
)
from scufris_hostd import (  # noqa: E402
    ActionKind,
    AuditLog,
    FakeExecutor,
    FakeFiles,
    HostdEngine,
    HostdServer,
    Requester,
)

SECRET = "an-example-shared-secret"

# What the faked machine says. The helper reads the CURRENT state to build its
# preview and its reversal, which is why canned output is enough.
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
REVERSE_DEPS = ok_result("nginx.service\nmulti-user.target\n")
# Two paths nothing references any more. `nix path-info` is left unanswered on
# purpose: the preview then reports the count with the size UNKNOWN rather than
# as zero, which is the honest shape and the one worth showing.
DEAD_PATHS = ok_result(
    "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-hello-2.12\n"
    "/nix/store/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb-ripgrep-14.1\n"
)


def _engine(directory: Path, executor: FakeExecutor) -> HostdEngine:
    return HostdEngine(
        AuditLog(directory / "audit.jsonl", secrets=frozenset({SECRET})),
        runner=FakeRunner(
            results={
                "systemctl --system show": UNIT_SHOW,
                "systemctl list-dependencies": REVERSE_DEPS,
                "nix-store --gc --print-dead": DEAD_PATHS,
            }
        ),
        executor=executor,
        files=FakeFiles(),
    )


async def main() -> int:
    # A SHORT temporary directory: a unix socket path is capped near 108 bytes.
    directory = Path(tempfile.mkdtemp(prefix="hostctl-example-"))
    executor = FakeExecutor(output=[("stdout", "Restarting nginx.service...\n")])
    server = HostdServer(
        _engine(directory, executor), secret=SECRET, socket_path=directory / "h.sock"
    )
    await server.start()

    database = open_database(directory)
    # The client's own two tables, created from the metadata it declares. The
    # app runs Alembic; an example that did would be an app example.
    Base.metadata.create_all(
        database.engine,
        tables=[
            Base.metadata.tables[name] for name in ("host_action", "config_change")
        ],
    )
    supervisor = host_supervisor()
    approvals = HostApprovalService(
        hostd=HostdClient(directory / "h.sock", SECRET),
        actions=HostActionStore(database),
        supervisor=supervisor,
    )

    try:
        restart = await approvals.propose(
            ActionKind.UNIT_RESTART,
            {"unit": "nginx.service"},
            Requester(actor="agent:host", agent="host"),
        )
        print(f"1. proposed {restart.id}")
        print(render_action(restart))

        cleanup = await approvals.propose(
            ActionKind.GC_STORE, {}, Requester(actor="operator")
        )
        gate = cleanup.confirmation
        print(f"\n2. proposed {cleanup.id}: {gate.risk_label}")
        print(f"   undo: {gate.undo}")
        print(f"   one-way: {gate.one_way}, acknowledge with {gate.acknowledge!r}")

        try:
            await approvals.approve(cleanup.id, actor="operator:example")
        except ConfirmationRequired as refused:
            print(f"3. approving it unacknowledged is refused: {refused}")
        else:  # pragma: no cover - the example asserts this below
            raise AssertionError("a one-way action was approved without its token")
        denied = await approvals.deny(
            cleanup.id, actor="operator:example", reason="not now"
        )
        print(f"   denied instead: {denied.decision}")

        # The ordinary action IS approved, and the apply is a supervised run
        # whose events any surface can attach to. Here the surface is a print.
        record, run_id = await approvals.approve(restart.id, actor="operator:example")
        print(f"\n4. approved {record.id} -> run {run_id}")
        bus = supervisor.bus(run_id)
        assert bus is not None, "an approved action must have a live run"
        streamed: list[str] = []
        async for _seq, event in bus.subscribe():
            if isinstance(event, HostApplyOutput):
                streamed.append(event.text.strip())
            elif isinstance(event, HostApplyResult):
                print(f"   result: ok={event.result.ok} {event.result.outcome}")
                break
        print(f"   streamed: {streamed}")

        applied = await approvals.get(restart.id)
        print(f"   journal: {applied.decision} by {applied.decided_by}")

        records = await HostdClient(directory / "h.sock", SECRET).audit_tail(limit=10)
        print(f"\n5. audit: {[record.event.value for record in records]}")

        # The claims, asserted rather than narrated.
        assert executor.calls == [step.argv for step in restart.proposal.steps], (
            executor.calls
        )
        assert applied.result is not None and applied.result.ok, applied
        assert records, "an action was applied and the helper recorded nothing"
    finally:
        await supervisor.aclose()
        database.close()
        await server.aclose()
        shutil.rmtree(directory, ignore_errors=True)

    print("\nOK: propose -> preview -> approve -> apply -> audit, from the client.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
