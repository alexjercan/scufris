"""Probe a local `opencode serve` daemon's health.

Standalone de-risk probe for the opencode backend spike
(tasks/20260722-135520). Hits ``GET /global/health`` and prints the version
and healthy flag. Deliberately self-contained (only ``httpx``, no scufris
imports) - the reusable client lands with the backend in
tasks/20260722-135525 and its ``examples/`` port supersedes this one.

How to run
----------
Start the daemon first, pointed at the llama.cpp provider config::

    OPENCODE_CONFIG=examples/opencode/opencode.json \\
      opencode serve --port 4096

Then::

    python examples/opencode/check_health.py

Env overrides::

    OPENCODE_URL=http://127.0.0.1:4096      # daemon base url
    OPENCODE_SERVER_PASSWORD=hunter2        # HTTP Basic password (username "")
"""

from __future__ import annotations

import os
import sys

import httpx


def main() -> int:
    url = os.environ.get("OPENCODE_URL", "http://127.0.0.1:4096").rstrip("/")
    password = os.environ.get("OPENCODE_SERVER_PASSWORD") or None
    auth = httpx.BasicAuth("", password) if password else None

    print(f"url:      {url}")
    print(f"password: {'<set>' if password else '<unset>'}")
    print()

    try:
        resp = httpx.get(f"{url}/global/health", auth=auth, timeout=10.0)
    except httpx.RequestError as exc:
        print(f"ERROR: cannot reach opencode at {url}: {exc!r}")
        print()
        print("Is the daemon running? Try: opencode serve --port 4096")
        return 1

    if resp.status_code != 200:
        print(f"ERROR: /global/health returned HTTP {resp.status_code}")
        return 1

    body = resp.json()
    print(f"version:  {body.get('version')}")
    print(f"healthy:  {body.get('healthy')}")
    print()
    if body.get("healthy") is not True:
        print("NOT HEALTHY")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
