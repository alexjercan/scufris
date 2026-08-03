#!/usr/bin/env python
"""Print what approving a host action from Telegram actually looks like.

    python examples/telegram_approval.py            # a reversible restart: one tap
    python examples/telegram_approval.py --one-way  # a store collection: two taps

The manual question this feature has to answer is "is this clear enough to decide
from a phone" - and that question is about TEXT. So this script boots the real app
(its approval service, its allowlist, its audit) with a real `scufris-hostd` on a
real unix socket, stubs the Bot API with respx, and prints every message and button
in the order the operator would see them:

    the host agent proposes
      -> the chat gets the proposal, with its buttons
      -> the operator taps (twice, for something that cannot be undone)
      -> the message is edited to say what happened

Nothing here touches the machine: the host commands come from a `FakeRunner`
replaying this box's real output, and a `FakeExecutor` records what it was asked to
run instead of running it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# Run from a checkout without installing it.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import httpx  # noqa: E402
import respx  # noqa: E402
from test_host_actions import host_files, host_runner  # noqa: E402

from scufris.agent_store import HOST_AGENT_ID  # noqa: E402
from scufris.app import create_app  # noqa: E402
from scufris.auth import hash_password  # noqa: E402
from scufris.config import Settings  # noqa: E402
from scufris.enums import AuthPolicy, Backend  # noqa: E402
from scufris_hostd import (  # noqa: E402
    AuditLog,
    FakeExecutor,
    HostdEngine,
    HostdServer,
)

API = "https://api.telegram.org/botDEMO"
CHAT = 4242
PASSWORD = "correct horse battery staple"
SECRET = "example-hostd-secret"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def show_message(body: dict[str, Any], *, edited: bool = False) -> None:
    """Print one chat message the way a phone stacks it: the text, then its buttons."""
    tag = "EDITED" if edited else "MESSAGE"
    print(f"  [{tag} to chat {body.get('chat_id')}]")
    for line in str(body.get("text", "")).splitlines() or [""]:
        print(f"  | {line}")
    markup = body.get("reply_markup") or {}
    rows = markup.get("inline_keyboard")
    if rows:
        for row in rows:
            print("  | " + "   ".join(f"[ {button['text']} ]" for button in row))
    elif rows == []:
        print("  | (no buttons: there is nothing left to decide)")
    elif markup.get("force_reply"):
        print("  | (waiting for a reply to this message)")
    print()


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


async def _run(directory: Path, one_way: bool) -> int:
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
        telegram_bot_token="DEMO",
        telegram_allowed_chat_ids=[CHAT],
        _env_file=None,  # type: ignore[call-arg]
    )
    app = create_app(settings=settings)

    sent: list[dict[str, Any]] = []
    edited: list[dict[str, Any]] = []
    toasts: list[str] = []
    counter = {"n": 100}

    def send(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        counter["n"] += 1
        return httpx.Response(
            200, json={"ok": True, "result": {"message_id": counter["n"]}}
        )

    def edit(request: httpx.Request) -> httpx.Response:
        edited.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "result": {}})

    def answer(request: httpx.Request) -> httpx.Response:
        toasts.append(str(json.loads(request.content).get("text", "")))
        return httpx.Response(200, json={"ok": True, "result": True})

    try:
        with respx.mock:
            # The poll loop is not what this demo is about: one 500 and it backs off.
            respx.post(f"{API}/getUpdates").mock(
                return_value=httpx.Response(500, json={"ok": False})
            )
            respx.post(f"{API}/sendMessage").mock(side_effect=send)
            respx.post(f"{API}/editMessageText").mock(side_effect=edit)
            respx.post(f"{API}/answerCallbackQuery").mock(side_effect=answer)

            async with app.router.lifespan_context(app):
                bot = app.state.telegram_bot
                if bot is None:
                    print("FAIL: no telegram bot was started")
                    return 1

                banner("1. the host agent proposes (nothing has happened yet)")
                kind = "gc_store" if one_way else "unit_restart"
                args: dict[str, object] = {} if one_way else {"unit": "nginx"}
                # Straight through the app's own objects: the helper mints the
                # proposal, the approval service records it, and the record_proposal
                # hook is what pushes it into the chat.
                proposal = await app.state.hostd.propose(kind, args, _requester())
                app.state.host_approvals.record_proposal(proposal)
                action_id = proposal.id
                await _settle_tasks()
                print(f"  (the executor has run {len(executor.calls)} commands)")

                banner("2. what arrives on the phone")
                for body in sent:
                    show_message(body)

                banner("3. the operator taps Approve")
                await bot._handle_update(_tap(action_id, "ha"))
                await _settle_tasks()
                if toasts:
                    print(f"  (the button answers: {toasts[-1]})")
                for body in edited:
                    show_message(body, edited=True)

                if one_way:
                    banner("4. it cannot be undone, so the second tap is the real one")
                    edited.clear()
                    await bot._handle_update(_tap(action_id, "hk"))
                    await _settle_tasks()
                    if toasts:
                        print(f"  (the button answers: {toasts[-1]})")

                banner("5. the message becomes the record of what happened")
                for _ in range(200):
                    if any("RESULT" in str(b.get("text", "")) for b in edited):
                        break
                    await asyncio.sleep(0.02)
                if edited:
                    show_message(edited[-1], edited=True)
                print(f"  (the executor has run {len(executor.calls)} commands)")

                banner("the record the helper wrote (root-owned in production)")
                for entry in audit.tail(20):
                    argv = (
                        " ".join(entry.steps[0].argv) if entry.steps else entry.detail
                    )
                    print(
                        f"{entry.at}  {entry.event:<10} "
                        f"{entry.requester.actor:<28} {argv}"
                    )
        return 0
    finally:
        asyncio.run_coroutine_threadsafe(server.aclose(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()
        shutil.rmtree(directory, ignore_errors=True)


def _requester() -> Any:
    """The host agent asking, as its MCP tool subprocess would be recorded."""
    from scufris_hostd import Requester

    return Requester(actor="agent", agent=HOST_AGENT_ID, run="run-1")


def _tap(action_id: str, verb: str) -> dict[str, Any]:
    return {
        "update_id": 1,
        "callback_query": {
            "id": f"cb-{verb}",
            "data": f"{verb}:{action_id}",
            "message": {"message_id": 101, "chat": {"id": CHAT}},
        },
    }


async def _settle_tasks() -> None:
    """Let the fire-and-forget announcements and the supervised apply run."""
    for _ in range(20):
        await asyncio.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--one-way",
        action="store_true",
        help="propose a store collection (no undo), which needs two taps",
    )
    args = parser.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="telegram-approval-example-"))
    started = time.monotonic()
    code = asyncio.run(_run(directory, args.one_way))
    print(f"\n({time.monotonic() - started:.1f}s)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
