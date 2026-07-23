"""Replay the stalled-merge scenario end to end and watch the loop self-heal.

This is the acceptance demo for bidirectional agent<->orchestrator comms
(spike 20260723-001256, BC5). It boots the real FastAPI app in-process against
the in-repo mock backend (no codex, no network, no browser) and drives the whole
loop over the app's own HTTP API - the exact endpoints the scufris MCP tools call:

    sub-agent blocks (request_input)  ->  a durable WAITING outcome
      ->  the orchestrator finds it (pending_agents, the poll path)
      ->  the orchestrator answers by resuming the sub-agent's session (message_agent -> chat)
      ->  the orchestrator clears it (acknowledge)  ->  the loop is resolved.

Because the mock backend does not run real MCP tools, each tool call is stood in
by a direct request to the endpoint that tool POSTs/GETs - which is exactly the
contract under test. The companion pytest acceptance
(`test_stalled_merge_loop_self_heals`) drives the same loop and also exercises the
`auto_wake` bridge path; this script is the human-readable walkthrough.

How to run
----------
    python examples/comms_loop.py

Self-contained: only needs scufris and httpx (both dev deps). Prints each step
and exits 0 when the loop resolves, 1 otherwise.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import httpx

from scufris.app import create_app
from scufris.config import Settings
from scufris.enums import Backend

QUESTION = "should I merge to master?"
ANSWER = "yes, merge it"


def _reply_from_sse(body: str) -> str:
    """Pull the assistant reply out of a buffered SSE chat response (the same
    `data: {...}` frames message_agent parses)."""
    reply = ""
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[len("data:") :].strip())
        except ValueError:
            continue
        if event.get("kind") == "done":
            reply = (event.get("reply") or {}).get("text", "") or reply
    return reply


async def run() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        proj = tmp_path / "proj"
        proj.mkdir()
        settings = Settings(
            web_dist=tmp_path / "absent",
            state_dir=tmp_path,
            agent_backend=Backend.MOCK,
            enable_mock_backend=True,
            # Poll path: the orchestrator finds the blocked agent itself. The
            # bridge (auto_wake=True) is the push alternative, covered by the test.
            auto_wake=False,
        )
        app = create_app(settings=settings)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://loop") as ac:
            await ac.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
            created = await ac.post(
                "/api/agents",
                json={
                    "name": "Builder",
                    "project_id": "my-app",
                    "backend": "mock",
                    "goal": "ship the feature",
                },
            )
            agent_id = created.json()["id"]
            print(f"1. launched sub-agent {agent_id!r} with goal 'ship the feature'")

            # The sub-agent hits a decision it cannot safely make and signals it.
            # In production its request_input MCP tool POSTs exactly this.
            r = await ac.post(
                f"/api/agents/{agent_id}/request_input", json={"question": QUESTION}
            )
            assert r.status_code == 200, r.text
            print(
                f"2. sub-agent BLOCKED, asked: {QUESTION!r}  ->  state={r.json()['state']}"
            )

            # The orchestrator polls for sub-agents that need it (auto_wake off).
            pending = (await ac.get("/api/agents/pending")).json()
            assert len(pending) == 1 and pending[0]["agent_id"] == agent_id, pending
            print(
                f"3. orchestrator POLLED pending_agents -> [{agent_id}: {pending[0]['message']!r}]"
            )

            # The orchestrator answers by resuming the sub-agent's own session.
            chat = await ac.post(
                f"/api/agents/{agent_id}/chat", json={"message": ANSWER}
            )
            assert chat.status_code == 200, chat.text
            reply = _reply_from_sse(chat.text)
            print(f"4. orchestrator ANSWERED via message_agent: {ANSWER!r}")
            print(f"   sub-agent resumed and replied: {reply!r}")

            # The orchestrator clears the signal so it stops pending. Answering by
            # resume already replaces the WAITING outcome with the resumed run's
            # DONE, so acknowledge is an idempotent belt-and-suspenders here (it may
            # report False if the DONE already landed) - what matters is that the
            # agent is no longer pending either way.
            await ac.post(f"/api/agents/{agent_id}/acknowledge")
            after = (await ac.get("/api/agents/pending")).json()
            assert after == [], after
            print(f"5. orchestrator cleared the signal -> pending now empty: {after}")

    print("\nOK - the stalled-merge loop self-healed end to end.")
    return 0


def main() -> int:
    try:
        return asyncio.run(run())
    except AssertionError as exc:
        print(f"\nFAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
