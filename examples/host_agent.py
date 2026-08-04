#!/usr/bin/env python
"""Drive the host AGENT's round trip and print what each side actually sees.

    the agent proposes -> the operator decides -> the agent is resumed with it

    python examples/host_agent.py            # the operator denies, with a reason
    python examples/host_agent.py --approve  # the operator approves; it applies

This is the half `examples/host_action.py` cannot show. That one drives the
helper's contract directly; this one runs the REAL app - its routes, its auth
middleware, its approval service, a real ``scufris-hostd`` on a real unix socket
in a background thread - and shows the three things the round trip is for:

1. the agent proposes with the machine credential it really carries, and the
   preview comes back as the operator-facing text rather than as JSON;
2. the agent is left BLOCKED, and the ORCHESTRATOR is refused when it tries to
   answer that - only the operator can, and this prints the refusal;
3. the decision resumes the agent's session with the outcome, INCLUDING the
   denial reason, which is the prompt printed at the end.

The only fake is the backend: turns are recorded instead of being sent to a
model, so the prompt the agent would receive is printed instead. The host
commands come from a ``FakeRunner`` replaying this box's real output and a
``FakeExecutor`` that records rather than runs, so nothing here touches the
machine.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator

# Run from a checkout without installing it.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "hostd" / "tests"))

from fastapi.testclient import TestClient  # noqa: E402
from test_host_actions import host_files, host_runner  # noqa: E402

from scufris.agent import AgentReply, StreamDone, StreamEvent  # noqa: E402
from scufris.agent_store import HOST_AGENT_ID  # noqa: E402
from scufris.app import create_app  # noqa: E402
from scufris.auth import CSRF_HEADER, hash_password  # noqa: E402
from scufris.backends import Capability  # noqa: E402
from scufris.config import Settings  # noqa: E402
from scufris.enums import AuthPolicy, Backend  # noqa: E402
from scufris.sessions import MemoryFootprint, UsageQuota  # noqa: E402
from scufris_hostctl import render_action  # noqa: E402
from scufris_hostd import (  # noqa: E402
    AuditLog,
    FakeExecutor,
    HostdEngine,
    HostdServer,
)

PASSWORD = "correct horse battery staple"
ORIGIN = "http://testserver"
SECRET = "example-hostd-secret"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


class RecordingBackend:
    """A backend that prints and records the prompt of every turn.

    The point of the example is the TEXT each side receives, so this stands where
    the model would be and shows what it would have been handed.
    """

    name = "recording"
    # The example never reads usage/memory, but the protocol makes every adapter
    # answer - which is the point of the seam.
    has_scufris_mcp = True

    def __init__(self) -> None:
        self.turns: list[tuple[str, str]] = []

    async def stream(
        self,
        settings: Settings,
        prompt: str,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        image_paths: list[str] | None = None,
        permission_mode: str = "manual",
        is_orchestrator: bool = False,
        agent_id: str = "",
    ) -> AsyncIterator[StreamEvent]:
        self.turns.append((agent_id, prompt))
        yield StreamDone(
            reply=AgentReply(text="(the agent's reply)", status="completed"),
            session_id=session_id or "host-session",
        )

    def read_status(self, settings: Settings, session_id: str | None) -> None:
        return None

    def read_transcript(self, settings: Settings, session_id: str | None) -> list[Any]:
        return []

    def read_context(self, settings: Settings, session_id: str | None) -> None:
        return None

    def read_usage(self, settings: Settings) -> Capability[UsageQuota]:
        return Capability.unsupported()

    def read_memory_footprint(self, settings: Settings) -> Capability[MemoryFootprint]:
        return Capability.unsupported()

    async def delete_session(self, settings: Settings, session_id: str | None) -> bool:
        return False


def start_helper(directory: Path) -> tuple[Path, FakeExecutor, threading.Thread, Any]:
    """A real hostd on a real socket, in its own event loop and thread."""
    audit = AuditLog(directory / "audit.jsonl", secrets=frozenset({SECRET}))
    executor = FakeExecutor()
    engine = HostdEngine(
        audit, runner=host_runner(), executor=executor, files=host_files()
    )
    socket_path = directory / "h.sock"
    server = HostdServer(engine, secret=SECRET, socket_path=socket_path)
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True, name="hostd-example")
    thread.start()
    if not ready.wait(timeout=10):
        raise SystemExit("the helper did not start")
    return socket_path, executor, thread, (loop, server, audit)


def wait_for_turn(backend: RecordingBackend, agent_id: str) -> str:
    """The prompt of the next turn launched for ``agent_id``.

    A decision launches that turn in the background, so this waits for it rather
    than assuming the HTTP response implied it.
    """
    for _ in range(300):
        for launched, prompt in backend.turns:
            if launched == agent_id:
                return prompt
        time.sleep(0.02)
    raise SystemExit(f"no turn was launched for {agent_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve",
        action="store_true",
        help="approve the action instead of denying it",
    )
    args = parser.parse_args()

    directory = Path(tempfile.mkdtemp(prefix="host-agent-example-"))
    socket_path, executor, thread, (loop, server, audit) = start_helper(directory)
    settings = Settings(
        state_dir=directory / "state",
        web_dist=directory / "absent",
        auth_mode=AuthPolicy.REQUIRED,
        auth_password_hash=hash_password(PASSWORD),
        hostd_socket=socket_path,
        hostd_secret=SECRET,
        agent_backend=Backend.MOCK,
        enable_mock_backend=True,
        _env_file=None,  # type: ignore[call-arg]
    )
    # Stand in for the model: every turn is recorded and printed instead of being
    # sent anywhere, which is how this example can show the exact prompt the agent
    # is resumed with.
    backend = RecordingBackend()
    import scufris.orchestrator.runs as runs_module

    # The bind site that LAUNCHES a turn: `get_backend` is imported by name, so
    # patching `scufris.backends` would not rebind this importer. `tests/conftest.py`
    # lists the same site.
    runs_module.get_backend = lambda _name: backend  # type: ignore[assignment]

    try:
        with TestClient(create_app(settings=settings)) as operator:
            # A separate client for the agent: it holds the machine bearer token and
            # no session cookie, which is exactly what its MCP subprocess has.
            agent = TestClient(operator.app)  # type: ignore[attr-defined]
            token = operator.app.state.api_token  # type: ignore[attr-defined]
            machine = {"Authorization": f"Bearer {token}", "Origin": ORIGIN}

            banner("1. the host agent proposes - and nothing has happened")
            proposed = agent.post(
                "/api/host/actions",
                json={
                    "kind": "unit_restart",
                    "args": {"unit": "nginx"},
                    "agent": HOST_AGENT_ID,
                },
                headers=machine,
            )
            if proposed.status_code != 201:
                print(f"the proposal was refused: {proposed.text}")
                return 1
            action_id = proposed.json()["proposal"]["id"]
            print(render_action(_record(proposed.json())))
            print(f"\n(the executor has run {len(executor.calls)} commands)")

            login = operator.post(
                "/api/auth/login",
                json={"password": PASSWORD},
                headers={"Origin": ORIGIN},
            )
            if login.status_code != 200:
                print(f"could not log in: {login.text}")
                return 1
            csrf = operator.cookies["scufris_csrf"]
            headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}

            banner("2. what the agent's state looks like while it waits")
            pending = operator.get("/api/agents/pending").json()
            for row in pending:
                print(f"  {row['agent_id']:<12} {row['state']:<8} {row['message']}")
            confirmation = operator.get(
                f"/api/host/actions/{action_id}/confirmation"
            ).json()
            print(f"\n  risk:      {confirmation['risk_label']}")
            print(f"  undo:      {confirmation['undo']}")
            print(f"  confirm:   {confirmation['style']}", end="")
            if confirmation["acknowledge"]:
                print(f" (must acknowledge {confirmation['acknowledge']!r})")
            else:
                print()

            banner("3. the ORCHESTRATOR cannot answer this - only the operator can")
            answered = agent.post(
                f"/api/agents/{HOST_AGENT_ID}/chat",
                json={"message": "approved, go ahead"},
                headers=machine,
            )
            print(f"  HTTP {answered.status_code}: {answered.json().get('detail', '')}")

            if args.approve:
                banner("4. the operator approves - THIS is what runs it")
                body = (
                    {"acknowledge": confirmation["acknowledge"]}
                    if confirmation["acknowledge"]
                    else {}
                )
                decided = operator.post(
                    f"/api/host/actions/{action_id}/approve",
                    headers=headers,
                    json=body,
                )
            else:
                banner("4. the operator says no, with a reason")
                decided = operator.post(
                    f"/api/host/actions/{action_id}/deny",
                    headers=headers,
                    json={
                        "reason": "nginx is serving the demo right now; "
                        "ask me after 18:00"
                    },
                )
            print(f"  HTTP {decided.status_code}")

            banner("5. the turn the agent is resumed with")
            print(wait_for_turn(backend, HOST_AGENT_ID))
            print(f"\n(the executor has run {len(executor.calls)} commands)")

            banner("the record the helper wrote (root-owned in production)")
            for entry in audit.tail(20):
                argv = " ".join(entry.steps[0].argv) if entry.steps else entry.detail
                print(
                    f"{entry.at}  {entry.event:<10} {entry.requester.actor:<24} {argv}"
                )
        return 0
    finally:
        asyncio.run_coroutine_threadsafe(server.aclose(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()
        shutil.rmtree(directory, ignore_errors=True)


def _record(payload: dict[str, Any]) -> Any:
    from scufris_hostctl import HostActionRecord

    return HostActionRecord.model_validate(payload)


if __name__ == "__main__":
    raise SystemExit(main())
