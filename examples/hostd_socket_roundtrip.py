#!/usr/bin/env python
"""The privileged helper's whole contract, over a real socket, with no host.

    python examples/hostd_socket_roundtrip.py

`scufris-hostd` is the one package whose boundary is not an import rule: it runs
as a separate process, as root, and everything it will ever do is asked for in
newline-delimited JSON over a unix socket. Carving it into its own distribution
is only honest if that boundary is exercisable, so this script drives it the way
the app does - through a raw socket, one request per connection, reading frames
until a terminal one arrives:

    1. hello       - the protocol version and the verbs on offer
    2. propose     - a unit restart, answered with a PREVIEW of what it would do
    3. list_pending- the helper is the source of truth for what is approvable
    4. apply       - streamed output frames, then one result frame
    5. audit_tail  - the root-owned record of what happened

Nothing here needs root, a network, or a NixOS machine. The system under the
helper is a `FakeRunner` replaying canned command output and a `FakeExecutor`
scripting the apply, which are the same seams the test suite drives - so what is
real in this script is the part that matters: the socket, the frames, the
authentication on every frame, and the refusal to run anything that was not
previewed and approved.

The client below is deliberately hand-rolled rather than
`scufris_hostctl.client`:
an example that imported the app's client would prove the app agrees with
itself. This one speaks the wire.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Run from a checkout without installing it. Two members: the helper, and the
# read-only inspection it builds its previews from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "hostd" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "host" / "src"))

from scufris_host import FakeRunner, ok_result  # noqa: E402
from scufris_hostd import (  # noqa: E402
    AuditLog,
    FakeExecutor,
    FakeFiles,
    HostdEngine,
    HostdServer,
)

SECRET = "an-example-shared-secret"

# What `systemctl` says about the unit being restarted. The helper reads the
# CURRENT state to build its preview and its reversal ("it was running, so
# stopping it again undoes this"), which is why a canned host is enough.
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


class Client:
    """The wire, by hand: one connection per request, frames until a terminal one."""

    #: Frame types that END a response. Anything else (``output``) is streamed.
    TERMINAL = frozenset({"hello", "proposal", "pending", "result", "audit", "error"})

    def __init__(self, socket_path: Path) -> None:
        self._path = socket_path

    def request(self, verb: str, **fields: Any) -> list[dict[str, Any]]:
        """Send one request and collect every frame the helper answers with.

        The connection stays open for the whole exchange: the helper treats EOF
        from the caller as a cancellation, so a client that half-closed after
        writing would cancel its own apply.
        """
        frames: list[dict[str, Any]] = []
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(30.0)
            connection.connect(str(self._path))
            # Every frame carries the secret. There is no handshake to skip.
            payload = json.dumps({"verb": verb, "secret": SECRET, **fields})
            connection.sendall(payload.encode("utf-8") + b"\n")
            for line in _lines(connection):
                frame = json.loads(line)
                frames.append(frame)
                if frame.get("type") in self.TERMINAL:
                    break
        return frames


def _lines(connection: socket.socket) -> Iterator[bytes]:
    """Newline-delimited JSON, without buffering a whole response in memory."""
    buffer = b""
    while True:
        chunk = connection.recv(65536)
        if not chunk:
            if buffer.strip():
                yield buffer
            return
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if line.strip():
                yield line


def _serve(server: HostdServer, ready: threading.Event) -> asyncio.AbstractEventLoop:
    """Run the helper in its own thread and event loop, as its own process would."""
    loop = asyncio.new_event_loop()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())
        ready.set()
        loop.run_forever()

    threading.Thread(target=run, daemon=True, name="hostd-example").start()
    assert ready.wait(timeout=10), "the helper did not start"
    return loop


def main() -> int:
    # A SHORT temporary directory: a unix socket path is capped near 108 bytes.
    directory = Path(tempfile.mkdtemp(prefix="hostd-example-"))
    audit = AuditLog(directory / "audit.jsonl", secrets=frozenset({SECRET}))
    executor = FakeExecutor(
        output=[("stdout", "Restarting nginx.service...\n")],
    )
    engine = HostdEngine(
        audit,
        runner=FakeRunner(
            results={
                "systemctl --system show": UNIT_SHOW,
                "systemctl list-dependencies": REVERSE_DEPS,
            }
        ),
        executor=executor,
        files=FakeFiles(),
    )
    server = HostdServer(engine, secret=SECRET, socket_path=directory / "h.sock")
    loop = _serve(server, threading.Event())
    client = Client(directory / "h.sock")

    try:
        hello = client.request("hello")[-1]
        print(f"1. hello: protocol v{hello['version']}, {len(hello['verbs'])} verbs")

        proposal = client.request(
            "propose",
            kind="unit_restart",
            args={"unit": "nginx.service"},
            requester={"actor": "operator"},
        )[-1]["proposal"]
        action_id = proposal["id"]
        argv = [" ".join(step["argv"]) for step in proposal["steps"]]
        print(f"2. proposed {action_id}: risk {proposal['risk']}, {argv}")
        print(f"   preview: {proposal['preview']['headline']}")
        print(f"   reversal: {proposal['reversal']['summary']}")

        pending = client.request("list_pending")[-1]["proposals"]
        print(f"3. pending: {[held['id'] for held in pending]}")

        frames = client.request(
            "apply", proposal_id=action_id, approved_by="operator:example"
        )
        streamed = [
            frame["text"].strip() for frame in frames if frame["type"] == "output"
        ]
        result = frames[-1]
        print(f"4. applied: ok={result['ok']} outcome={result['outcome']} {streamed}")

        records = client.request("audit_tail", limit=10)[-1]["records"]
        print(f"5. audit: {[record['event'] for record in records]}")

        # The claim, asserted rather than narrated: the helper ran exactly the
        # argv the operator approved, and wrote a record of doing so.
        assert executor.calls == [step["argv"] for step in proposal["steps"]], (
            executor.calls
        )
        assert result["ok"] is True, result
        assert records, "the helper applied an action and recorded nothing"
    finally:
        asyncio.run_coroutine_threadsafe(server.aclose(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)

    print("\nOK: propose -> preview -> approve -> apply -> audit, over a socket.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
