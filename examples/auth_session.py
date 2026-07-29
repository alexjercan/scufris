"""Walk the dashboard's authentication boundary end to end, over a real socket.

This is the human-readable companion to `tests/test_auth.py`. It boots the real
FastAPI app with authentication REQUIRED, on a real uvicorn port, and drives the
whole boundary with an HTTP client - no browser, no codex, no network:

    denied (no session)  ->  wrong password (and the throttle counting)
      ->  login  ->  the session works  ->  a state change without the CSRF token
      is still refused  ->  with it, it passes  ->  a cross-origin attempt fails
      ->  the machine (MCP tool) token works  ->  logout revokes the session.

Why it exists: "the dashboard has a login" is easy to believe and hard to check.
Run this and watch each refusal happen for its own reason. See
`tasks/20260729-125015/DECISION.md` for why the mechanism is shaped this way.

How to run
----------
    python examples/auth_session.py

Self-contained: only needs scufris, httpx and uvicorn (all dev deps). Prints each
step and exits 0 when every check holds, 1 otherwise.
"""

from __future__ import annotations

import socket
import tempfile
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from scufris.app import create_app
from scufris.auth import CSRF_COOKIE, CSRF_HEADER, SESSION_COOKIE, hash_password
from scufris.config import Settings
from scufris.enums import AuthPolicy

PASSWORD = "the operator's password"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{f' - {detail}' if detail else ''}")
    if not ok:
        failures.append(label)


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def main() -> int:
    state = Path(tempfile.mkdtemp(prefix="scufris-auth-example-"))
    port = free_port()
    settings = Settings(
        host="127.0.0.1",
        port=port,
        web_dist=state / "absent",
        state_dir=state,
        # REQUIRED rather than the loopback default, so this example exercises the
        # deployed posture rather than the development one.
        auth_mode=AuthPolicy.REQUIRED,
        auth_password_hash=hash_password(PASSWORD),
        agent_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    app = create_app(settings=settings)
    machine_token = app.state.api_token

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        if not server.started:
            print("uvicorn did not start")
            return 1

        base = f"http://127.0.0.1:{port}"
        origin = {"Origin": base}
        client = httpx.Client(base_url=base, timeout=10)

        print("\n1. with no session, the dashboard answers nothing")
        resp = client.get("/api/stats")
        check(
            "GET /api/stats is refused", resp.status_code == 401, str(resp.status_code)
        )
        resp = client.get("/api/auth/session")
        check(
            "the session probe is public and honest",
            resp.status_code == 200
            and resp.json() == {"authenticated": False, "required": True},
            resp.text.strip(),
        )

        print("\n2. a wrong password is refused, and counted")
        resp = client.post("/api/auth/login", json={"password": "nope"}, headers=origin)
        check("wrong password -> 401", resp.status_code == 401, resp.text.strip())
        check("no session cookie was issued", SESSION_COOKIE not in resp.cookies)

        print("\n3. the right password opens a session")
        resp = client.post(
            "/api/auth/login", json={"password": PASSWORD}, headers=origin
        )
        check("login -> 200", resp.status_code == 200)
        csrf = client.cookies.get(CSRF_COOKIE) or ""
        check("a session cookie was issued", bool(client.cookies.get(SESSION_COOKIE)))
        check("a CSRF token was issued", bool(csrf))
        check("GET /api/stats now works", client.get("/api/stats").status_code == 200)

        print("\n4. a session alone does not authorize a state change")
        resp = client.post("/api/chat/reset", headers=origin)
        check("no CSRF header -> 403", resp.status_code == 403, resp.text.strip())
        resp = client.post("/api/chat/reset", headers={**origin, CSRF_HEADER: csrf})
        check(
            "with the CSRF header it passes the gate",
            resp.status_code not in (401, 403),
            str(resp.status_code),
        )

        print("\n5. another site cannot ride the session")
        resp = client.post(
            "/api/chat/reset",
            headers={"Origin": "http://evil.example", CSRF_HEADER: csrf},
        )
        check("cross-origin -> 403", resp.status_code == 403, resp.text.strip())

        print("\n6. the app's own tool subprocesses use a machine token, not a cookie")
        bare = httpx.Client(base_url=base, timeout=10)
        check(
            "no credential -> 401",
            bare.get("/api/projects").status_code == 401,
        )
        check(
            "the machine token is accepted",
            bare.get(
                "/api/projects", headers={"Authorization": f"Bearer {machine_token}"}
            ).status_code
            == 200,
        )
        check(
            "a forged token is not",
            bare.get(
                "/api/projects", headers={"Authorization": "Bearer not-the-token"}
            ).status_code
            == 401,
        )

        print("\n7. logout revokes the session server-side")
        resp = client.post("/api/auth/logout", headers={**origin, CSRF_HEADER: csrf})
        check("logout -> 200", resp.status_code == 200)
        check("the session is dead", client.get("/api/stats").status_code == 401)
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all checks held: the boundary refuses, permits, and revokes as designed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
