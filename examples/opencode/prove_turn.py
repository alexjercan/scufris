"""Prove one end-to-end turn through `opencode serve` against llama.cpp.

Standalone de-risk probe for the opencode backend spike
(tasks/20260722-135520). Creates a session, sends one user message routed to
the local llama.cpp provider, and prints the assistant's reply - the concrete
proof that scufris -> opencode serve -> llama-server works before the backend
(tasks/20260722-135525) wraps it. Self-contained (only ``httpx``).

The turn is SYNCHRONOUS: ``POST /session/:id/message`` blocks until the model
finishes, then returns the whole ``{info, parts}`` payload. This is the shape
the v0 backend drives (live token streaming over ``/event`` is a later
follow-up).

How to run
----------
Start the daemon first::

    OPENCODE_CONFIG=examples/opencode/opencode.json \\
      opencode serve --port 4096

Then::

    python examples/opencode/prove_turn.py "say hi in five words"

Env overrides::

    OPENCODE_URL=http://127.0.0.1:4096
    OPENCODE_SERVER_PASSWORD=hunter2
    OPENCODE_PROVIDER=llamacpp
    OPENCODE_MODEL=gemma-4-26B-A4B-it
    OPENCODE_TIMEOUT=600            # seconds; a cold model load is slow

Note: the first turn against a model that is not resident triggers a
llama-server load (and a HuggingFace download on first ever use), which can
take many minutes. Keep OPENCODE_TIMEOUT generous.
"""

from __future__ import annotations

import os
import sys

import httpx


def main(argv: list[str]) -> int:
    url = os.environ.get("OPENCODE_URL", "http://127.0.0.1:4096").rstrip("/")
    password = os.environ.get("OPENCODE_SERVER_PASSWORD") or None
    provider = os.environ.get("OPENCODE_PROVIDER", "llamacpp")
    model = os.environ.get("OPENCODE_MODEL", "gemma-4-26B-A4B-it")
    timeout = float(os.environ.get("OPENCODE_TIMEOUT", "600"))
    prompt = " ".join(argv[1:]) or "Reply with exactly: hello from gemma"

    auth = httpx.BasicAuth("", password) if password else None
    print(f"url:      {url}")
    print(f"model:    {provider}/{model}")
    print(f"prompt:   {prompt!r}")
    print()

    with httpx.Client(base_url=url, auth=auth, timeout=timeout) as client:
        session = client.post("/session", json={"title": "scufris probe"})
        session.raise_for_status()
        session_id = session.json()["id"]
        print(f"session:  {session_id}")

        resp = client.post(
            f"/session/{session_id}/message",
            json={
                "model": {"providerID": provider, "modelID": model},
                "parts": [{"type": "text", "text": prompt}],
            },
        )
    resp.raise_for_status()
    body = resp.json()

    parts = body.get("parts", [])
    text = "".join(p.get("text") or "" for p in parts if p.get("type") == "text")
    tool_parts = [p for p in parts if "tool" in str(p.get("type", ""))]

    print(f"modelID:  {body.get('info', {}).get('modelID')}")
    print(f"tools:    {len(tool_parts)} tool part(s)")
    print()
    print("reply:")
    print(text or "(no text parts - tool-only or empty turn)")
    print()
    if not text.strip():
        print("FAIL: assistant produced no text")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
