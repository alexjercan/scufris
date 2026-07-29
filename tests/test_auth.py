"""Tests for the dashboard's authenticated session boundary.

The load-bearing test here is ``test_authenticated_session_and_csrf_boundary``:
it derives the protected surface from ``app.routes`` rather than a hand-written
list, so a route added later is covered by existing rather than by remembering
to add it. See ``tasks/20260729-125015/DECISION.md``.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Route

from scufris.app import create_app
from scufris.auth import (
    CSRF_COOKIE,
    CSRF_HEADER,
    SESSION_COOKIE,
    AuthConfigError,
    LoginThrottle,
    SessionStore,
    hash_password,
    verify_password,
)
from scufris.config import Settings
from scufris.enums import AuthPolicy, Backend
from scufris.metrics import Collector


async def _collect(events: Any) -> list[Any]:
    """Drain an async event stream into a list."""
    return [event async for event in events]


PASSWORD = "correct horse battery staple"
ORIGIN = "http://testserver"


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    """A hermetic Settings with authentication configured.

    ``_env_file=None`` so a developer's real ``.env`` cannot leak in (lesson
    settings-test-must-disable-env-file)."""
    base: dict[str, Any] = {
        "web_dist": tmp_path / "absent",
        "state_dir": tmp_path,
        "auth_mode": AuthPolicy.REQUIRED,
        "auth_password_hash": hash_password(PASSWORD),
        "_env_file": None,
    }
    base.update(kwargs)
    return Settings(**base)


def _login(client: TestClient, password: str = PASSWORD) -> str:
    """Log in and return the CSRF token the server issued for the session."""
    resp = client.post(
        "/api/auth/login", json={"password": password}, headers={"Origin": ORIGIN}
    )
    assert resp.status_code == 200, resp.text
    token = client.cookies.get(CSRF_COOKIE)
    assert token, "login did not issue a CSRF cookie"
    return token


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --- the password hash ------------------------------------------------------


def test_password_hash_round_trips_and_rejects_a_wrong_password() -> None:
    encoded = hash_password(PASSWORD)
    assert encoded.startswith("scrypt$")
    assert verify_password(PASSWORD, encoded)
    assert not verify_password(PASSWORD + "x", encoded)
    assert not verify_password("", encoded)
    # Salted: the same password hashes differently every time.
    assert hash_password(PASSWORD) != encoded


def test_verify_password_rejects_a_malformed_hash() -> None:
    """A corrupt or truncated hash must fail closed, never raise into a 500 and
    never authenticate."""
    for bad in ("", "scrypt$", "scrypt$notanint$8$1$aaaa$bbbb", "plaintext", "$$$$"):
        assert not verify_password(PASSWORD, bad)


# --- the enforcement boundary (DoD) -----------------------------------------


def _methods(route: Route) -> set[str]:
    """The route's HTTP methods (the attribute is typed optional upstream)."""
    return set(route.methods or ())


def _concrete_path(path: str) -> str:
    """Substitute a dummy value for every path parameter."""
    out: list[str] = []
    for segment in path.split("/"):
        out.append("x" if segment.startswith("{") else segment)
    return "/".join(out)


def test_authenticated_session_and_csrf_boundary(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """Every API route except the public allowlist rejects an unauthenticated
    request, and every state-changing route rejects a session without a matching
    CSRF header.

    The surface is ENUMERATED from ``app.routes`` so a new route is covered
    without touching this test."""
    app = create_app(collector=fake_collector, settings=_settings(tmp_path))
    from scufris.auth import PUBLIC_PATHS, PUBLIC_STATIC_PATHS

    client = TestClient(app)

    checked_unauth = 0
    checked_csrf = 0
    for route in app.routes:
        # Every Route, not just APIRoute: /openapi.json, /docs and /redoc are
        # plain starlette Routes and would otherwise sit outside this alarm.
        if not isinstance(route, Route):
            continue
        if route.path in PUBLIC_PATHS or route.path in PUBLIC_STATIC_PATHS:
            continue
        target = _concrete_path(route.path)
        for method in sorted(_methods(route) - {"HEAD", "OPTIONS"}):
            resp = client.request(method, target, headers={"Origin": ORIGIN})
            assert resp.status_code == 401, (
                f"{method} {target} answered {resp.status_code} with no session; "
                "every non-public route must be gated"
            )
            checked_unauth += 1

    # Now authenticated: a state-changing request without the CSRF header is
    # still refused, and the same request with it passes the gate.
    csrf = _login(client)
    for route in app.routes:
        if not isinstance(route, Route):
            continue
        if route.path in PUBLIC_PATHS or route.path in PUBLIC_STATIC_PATHS:
            continue
        target = _concrete_path(route.path)
        for method in sorted(_methods(route) - {"HEAD", "OPTIONS", "GET"}):
            resp = client.request(method, target, headers={"Origin": ORIGIN})
            assert resp.status_code == 403, (
                f"{method} {target} answered {resp.status_code} with a session but "
                "no CSRF header; state-changing routes must require it"
            )
            # With the token the gate lets it through to the handler, which may
            # legitimately 404/422/400 on the dummy path - anything but the gate's
            # own 401/403.
            passed = client.request(
                method, target, headers={"Origin": ORIGIN, CSRF_HEADER: csrf}
            )
            assert passed.status_code not in (401, 403), (
                f"{method} {target} was refused {passed.status_code} WITH a valid "
                "session and CSRF token"
            )
            checked_csrf += 1
            # Logout is in the swept surface (it is not public), and calling it
            # successfully ends the session the rest of the sweep is using.
            if target == "/api/auth/logout":
                csrf = _login(client)

    # Guard the guard: if the sweep ever selects nothing it must fail loudly
    # rather than pass vacuously.
    assert checked_unauth > 40, f"sweep covered only {checked_unauth} route/methods"
    assert checked_csrf > 10, f"CSRF sweep covered only {checked_csrf} route/methods"


def test_public_paths_are_reachable_without_a_session(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The login endpoint and the session probe must answer unauthenticated, or
    the login page cannot function."""
    client = TestClient(
        create_app(collector=fake_collector, settings=_settings(tmp_path))
    )

    probe = client.get("/api/auth/session")
    assert probe.status_code == 200
    assert probe.json()["authenticated"] is False

    bad = client.post(
        "/api/auth/login", json={"password": "wrong"}, headers={"Origin": ORIGIN}
    )
    assert bad.status_code == 401
    assert SESSION_COOKIE not in bad.cookies


def test_login_logout_round_trip(fake_collector: Collector, tmp_path: Path) -> None:
    app = create_app(collector=fake_collector, settings=_settings(tmp_path))
    client = TestClient(app)

    assert client.get("/api/stats").status_code == 401
    csrf = _login(client)
    assert client.get("/api/stats").status_code == 200
    assert client.get("/api/auth/session").json()["authenticated"] is True

    out = client.post("/api/auth/logout", headers={"Origin": ORIGIN, CSRF_HEADER: csrf})
    assert out.status_code == 200
    assert client.get("/api/stats").status_code == 401


def test_login_rotates_the_session_id(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """Session fixation: a session id that existed before login must not be the
    one that carries the authenticated session afterwards, and the previous
    session is revoked rather than left live.

    Cookies are set through explicit ``Cookie`` headers rather than the client
    jar: planting one in the jar and then receiving the server's own makes httpx
    raise CookieConflict on read."""
    app = create_app(collector=fake_collector, settings=_settings(tmp_path))
    planted = "attacker-planted-session-id"

    client = TestClient(app)
    resp = client.post(
        "/api/auth/login",
        json={"password": PASSWORD},
        headers={"Origin": ORIGIN, "Cookie": f"{SESSION_COOKIE}={planted}"},
    )
    assert resp.status_code == 200
    first = resp.cookies.get(SESSION_COOKIE)
    assert first and first != planted

    # The planted id never becomes authenticated.
    fresh = TestClient(app)
    assert (
        fresh.get(
            "/api/stats", headers={"Cookie": f"{SESSION_COOKIE}={planted}"}
        ).status_code
        == 401
    )
    assert (
        fresh.get(
            "/api/stats", headers={"Cookie": f"{SESSION_COOKIE}={first}"}
        ).status_code
        == 200
    )

    # A second login rotates again AND revokes the first session.
    second_resp = TestClient(app).post(
        "/api/auth/login",
        json={"password": PASSWORD},
        headers={"Origin": ORIGIN, "Cookie": f"{SESSION_COOKIE}={first}"},
    )
    second = second_resp.cookies.get(SESSION_COOKIE)
    assert second and second != first
    assert (
        fresh.get(
            "/api/stats", headers={"Cookie": f"{SESSION_COOKIE}={first}"}
        ).status_code
        == 401
    ), "the previous session must be revoked on re-login, not left live"


def test_a_forged_session_id_is_refused(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_settings(tmp_path))
    )
    client.cookies.set(SESSION_COOKIE, "0" * 43)
    assert client.get("/api/stats").status_code == 401


def test_expired_session_is_refused(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An idle session past its TTL stops working without a restart."""
    settings = _settings(tmp_path, auth_session_idle_seconds=1.0)
    app = create_app(collector=fake_collector, settings=settings)
    client = TestClient(app)
    _login(client)
    assert client.get("/api/stats").status_code == 200

    now = time.time()
    monkeypatch.setattr("scufris.auth.time.time", lambda: now + 5.0)
    assert client.get("/api/stats").status_code == 401


def test_absolute_session_cap_expires_an_active_session(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sliding renewal must not outlive the absolute cap: a session that is used
    continuously still dies at the cap."""
    settings = _settings(
        tmp_path, auth_session_idle_seconds=1000.0, auth_session_max_seconds=10.0
    )
    app = create_app(collector=fake_collector, settings=settings)
    client = TestClient(app)
    _login(client)

    now = time.time()
    for offset in (2.0, 4.0, 6.0, 8.0):
        monkeypatch.setattr("scufris.auth.time.time", lambda o=offset: now + o)
        assert client.get("/api/stats").status_code == 200
    monkeypatch.setattr("scufris.auth.time.time", lambda: now + 11.0)
    assert client.get("/api/stats").status_code == 401


def test_cross_origin_state_change_is_refused(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """Even with a valid session AND a valid CSRF token, a request whose Origin
    is another site is refused."""
    app = create_app(collector=fake_collector, settings=_settings(tmp_path))
    client = TestClient(app)
    csrf = _login(client)

    evil = client.post(
        "/api/chat/reset",
        headers={"Origin": "http://evil.example", CSRF_HEADER: csrf},
    )
    assert evil.status_code == 403

    # A missing Origin AND Referer is refused too - a browser always sends one on
    # a state-changing request, so its absence is not a shape we trust.
    bare = client.post("/api/chat/reset", headers={CSRF_HEADER: csrf})
    assert bare.status_code == 403

    # Referer alone (some privacy settings strip Origin) is accepted when it
    # matches.
    ok = client.post(
        "/api/chat/reset",
        headers={"Referer": f"{ORIGIN}/", CSRF_HEADER: csrf},
    )
    assert ok.status_code not in (401, 403)


def test_csrf_token_from_another_session_is_refused(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The CSRF token is bound to the session, so a token harvested elsewhere
    does not satisfy the check."""
    app = create_app(collector=fake_collector, settings=_settings(tmp_path))
    other = TestClient(app)
    stolen = _login(other)

    client = TestClient(app)
    _login(client)
    resp = client.post(
        "/api/chat/reset", headers={"Origin": ORIGIN, CSRF_HEADER: stolen}
    )
    assert resp.status_code == 403


def test_failed_logins_are_throttled(fake_collector: Collector, tmp_path: Path) -> None:
    """Repeated wrong passwords lock the source out, and the lockout applies to
    the CORRECT password too - otherwise it is not a lockout."""
    settings = _settings(tmp_path, auth_login_max_failures=3)
    client = TestClient(create_app(collector=fake_collector, settings=settings))

    for _ in range(3):
        assert (
            client.post(
                "/api/auth/login", json={"password": "no"}, headers={"Origin": ORIGIN}
            ).status_code
            == 401
        )
    locked = client.post(
        "/api/auth/login", json={"password": PASSWORD}, headers={"Origin": ORIGIN}
    )
    assert locked.status_code == 429
    assert SESSION_COOKIE not in locked.cookies


def test_throttle_releases_after_the_window() -> None:
    throttle = LoginThrottle(max_failures=2, window_seconds=60.0)
    now = 1000.0
    assert throttle.allowed("1.2.3.4", now=now)
    throttle.record_failure("1.2.3.4", now=now)
    throttle.record_failure("1.2.3.4", now=now)
    assert not throttle.allowed("1.2.3.4", now=now)
    # A different source is unaffected by another's failures.
    assert throttle.allowed("5.6.7.8", now=now)
    assert throttle.allowed("1.2.3.4", now=now + 61.0)


def test_throttle_clears_on_success() -> None:
    throttle = LoginThrottle(max_failures=3, window_seconds=60.0)
    throttle.record_failure("1.2.3.4", now=1000.0)
    throttle.record_success("1.2.3.4")
    assert throttle.allowed("1.2.3.4", now=1000.0)


# --- the loopback / fail-closed policy (DoD) --------------------------------


def test_loopback_only_auth_policy(tmp_path: Path) -> None:
    """`auto` means: open on loopback, mandatory off it. That is what keeps
    pytest and the examples free of a login dance while the deployed bind is
    protected without opting in."""
    from scufris.auth import auth_required

    loopback = Settings(
        host="127.0.0.1",
        state_dir=tmp_path,
        _env_file=None,  # type: ignore[call-arg]
    )
    assert loopback.auth_mode is AuthPolicy.AUTO
    assert auth_required(loopback) is False

    for host in ("::1", "localhost"):
        assert (
            auth_required(
                Settings(host=host, state_dir=tmp_path, _env_file=None)  # type: ignore[call-arg]
            )
            is False
        )

    lan = Settings(
        host="0.0.0.0",
        state_dir=tmp_path,
        auth_password_hash=hash_password(PASSWORD),
        _env_file=None,  # type: ignore[call-arg]
    )
    assert auth_required(lan) is True

    # `required` forces it on even on loopback...
    forced = Settings(
        host="127.0.0.1",
        state_dir=tmp_path,
        auth_mode=AuthPolicy.REQUIRED,
        auth_password_hash=hash_password(PASSWORD),
        _env_file=None,  # type: ignore[call-arg]
    )
    assert auth_required(forced) is True

    # ...and `disabled` turns it off, but ONLY on loopback (see the refusal test).
    off = Settings(
        host="127.0.0.1",
        state_dir=tmp_path,
        auth_mode=AuthPolicy.DISABLED,
        _env_file=None,  # type: ignore[call-arg]
    )
    assert auth_required(off) is False


def test_non_loopback_bind_without_credentials_refuses_to_start(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """Fail closed: a LAN bind with no password hash must not serve at all. A
    warning-and-serve here would be the exact hole this task exists to close."""
    settings = Settings(
        host="0.0.0.0",
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        _env_file=None,  # type: ignore[call-arg]
    )
    with pytest.raises(AuthConfigError) as excinfo:
        create_app(collector=fake_collector, settings=settings)
    assert "SCUFRIS_AUTH_PASSWORD_HASH" in str(excinfo.value)

    # And `disabled` cannot be used to opt out of a non-loopback bind.
    opted_out = Settings(
        host="0.0.0.0",
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        auth_mode=AuthPolicy.DISABLED,
        auth_password_hash=hash_password(PASSWORD),
        _env_file=None,  # type: ignore[call-arg]
    )
    with pytest.raises(AuthConfigError) as disabled_exc:
        create_app(collector=fake_collector, settings=opted_out)
    assert "disabled" in str(disabled_exc.value).lower()

    # `required` with no credential is equally unservable, on any bind.
    no_secret = Settings(
        host="127.0.0.1",
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        auth_mode=AuthPolicy.REQUIRED,
        _env_file=None,  # type: ignore[call-arg]
    )
    with pytest.raises(AuthConfigError):
        create_app(collector=fake_collector, settings=no_secret)


def test_loopback_app_needs_no_login(fake_collector: Collector, tmp_path: Path) -> None:
    """The development shape: no credential, loopback bind, everything open."""
    settings = Settings(
        host="127.0.0.1",
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        _env_file=None,  # type: ignore[call-arg]
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    assert client.get("/api/stats").status_code == 200
    assert client.post("/api/chat/reset").status_code != 401


# --- the session store ------------------------------------------------------


def test_sessions_survive_a_restart(fake_collector: Collector, tmp_path: Path) -> None:
    """The session record is persisted, so restarting the server does not log the
    operator out."""
    settings = _settings(tmp_path)
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    _login(client)
    cookie = client.cookies.get(SESSION_COOKIE)

    fresh = TestClient(
        create_app(collector=fake_collector, settings=_settings(tmp_path))
    )
    fresh.cookies.set(SESSION_COOKIE, str(cookie))
    assert fresh.get("/api/stats").status_code == 200


def test_session_file_is_not_world_readable(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "auth_sessions.json")
    store.create(now=1000.0)
    path = tmp_path / "auth_sessions.json"
    assert path.exists()
    assert path.stat().st_mode & 0o077 == 0, (
        "session file must not be group/other readable"
    )


def test_revoke_all_invalidates_every_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "auth_sessions.json")
    a = store.create(now=1000.0)
    b = store.create(now=1000.0)
    store.revoke_all()
    assert store.get(a.id, now=1000.0, idle=100.0, absolute=100.0) is None
    assert store.get(b.id, now=1000.0, idle=100.0, absolute=100.0) is None


# --- machine callers (DoD) --------------------------------------------------


def test_mcp_tools_reach_the_api_under_auth(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MCP tool servers and the in-process operator console call this app's
    own HTTP API with no cookie. With authentication on they must still get
    through, via the per-process bearer token - and a caller WITHOUT the token
    must not.

    Driven over a REAL uvicorn socket because that is the shape the tool takes
    (a blocking httpx call looping back to this server); ASGITransport cannot
    exercise it. Mirrors test_app::test_tool_console_self_loopback."""
    import httpx
    import uvicorn

    from scufris import mcp_common

    port = _free_port()
    settings = _settings(tmp_path, port=port)
    app = create_app(collector=fake_collector, settings=settings)
    token = app.state.api_token
    assert token, "the app must mint a machine token"

    monkeypatch.setenv("SCUFRIS_API_BASE", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("SCUFRIS_API_TOKEN", token)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started, "uvicorn did not start"

        # The real tool helper, with the token in its environment.
        out = mcp_common._api_call("GET", "/api/projects")  # noqa: SLF001
        assert not out.startswith("error:"), out

        # A state-changing tool call needs no CSRF token: the bearer caller has no
        # ambient cookie to be ridden, and requiring one would break every tool.
        created = mcp_common._api_call(  # noqa: SLF001
            "POST", "/api/projects", body={"path": str(tmp_path)}
        )
        assert not created.startswith("error: 401"), created
        assert not created.startswith("error: 403"), created

        # Without the token the same call is refused - the loopback address alone
        # buys nothing.
        monkeypatch.delenv("SCUFRIS_API_TOKEN")
        denied = mcp_common._api_call("GET", "/api/projects")  # noqa: SLF001
        assert denied.startswith("error: 401"), denied

        # A wrong token is refused too.
        monkeypatch.setenv("SCUFRIS_API_TOKEN", "not-the-token")
        forged = mcp_common._api_call("GET", "/api/projects")  # noqa: SLF001
        assert forged.startswith("error: 401"), forged

        # The tool console runs the tool IN THIS PROCESS, and gets its credential
        # from the ContextVar the endpoint sets - NOT from the environment. Clear
        # the env var entirely so only that path can make this work.
        monkeypatch.delenv("SCUFRIS_API_TOKEN", raising=False)

        # And the tool console (in-process, on the server's own loop) still runs.
        resp = httpx.post(
            f"http://127.0.0.1:{port}/api/agent/tools/pending_agents/run",
            json={"args": {}},
            headers={"Authorization": f"Bearer {token}"},
            timeout=8,
        )
        assert resp.status_code == 200
        assert "no agents are waiting" in resp.json()["text"]
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_agent_env_carries_the_machine_token(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The token has to actually reach the MCP subprocess environment, or the
    tools authenticate with nothing. Asserts the wiring, not just the mint.

    Both audiences matter: the orchestrator's ``scufris`` server and a sub-agent's
    ``agent`` callback server BOTH call the API (lesson
    tool-reachable-by-two-runners-needs-a-test-per-runner).

    Note there is no environment seeding here: the token travels on the app's own
    Settings, so this is the real path. The companion check that it does NOT
    reach the agent CLI is
    ``test_agent_cli_env_does_not_carry_the_machine_token``."""
    from scufris.agent import scufris_mcp_servers

    settings = _settings(tmp_path, den_path=tmp_path / "den")
    app = create_app(collector=fake_collector, settings=settings)
    token = app.state.api_token

    orchestrator = scufris_mcp_servers(settings, is_orchestrator=True)
    by_name = {server.server_id: server.env for server in orchestrator}
    assert by_name["scufris"].get("SCUFRIS_API_TOKEN") == token

    sub_agent = scufris_mcp_servers(settings, agent_id="a1")
    assert sub_agent[0].env.get("SCUFRIS_API_TOKEN") == token

    # The den server does NOT call the API, so it has no business holding a
    # credential for it.
    assert "SCUFRIS_API_TOKEN" not in by_name["den"]


# --- streaming and the Telegram bridge (DoD) --------------------------------


class _FakeBot:
    """Stand-in for TelegramBot that records construction and never polls."""

    instances: list[_FakeBot] = []

    def __init__(
        self,
        token: str,
        allowed: Any,
        on_message: Any,
        on_reset: Any,
        on_cancel: Any,
        **kwargs: Any,
    ) -> None:
        self.token = token
        self.allowed = allowed
        self.on_message = on_message
        _FakeBot.instances.append(self)

    async def run(self) -> None:
        import asyncio

        await asyncio.Event().wait()


def test_authenticated_streaming_and_telegram_bridge(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With authentication ON: an SSE stream still streams for a logged-in
    operator and is refused without a session, and the Telegram bridge still
    starts and drives turns (its auth is the chat-id allowlist, not a cookie)."""
    _FakeBot.instances.clear()
    monkeypatch.setattr("scufris.app.TelegramBot", _FakeBot)
    settings = _settings(
        tmp_path,
        telegram_bot_token="TOKEN123",
        telegram_allowed_chat_ids=[100],
        agent_backend=Backend.MOCK,
        enable_mock_backend=True,
    )
    app = create_app(collector=fake_collector, settings=settings)

    with TestClient(app) as client:
        # The bridge came up under authentication.
        assert app.state.telegram_task is not None
        assert not app.state.telegram_task.done()
        assert len(_FakeBot.instances) == 1
        # ...and it is wired to a real turn callback, not gated behind a cookie.
        assert callable(_FakeBot.instances[0].on_message)

        # SSE without a session is refused - and refused by the GATE, before the
        # handler's own 404, so an unauthenticated caller cannot use status codes
        # to probe which agents exist.
        assert client.get("/api/agents/orchestrator/events").status_code == 401

        csrf = _login(client)
        # Authenticated, the same request reaches the handler (which 404s because
        # no run is live) rather than the gate.
        assert client.get("/api/agents/orchestrator/events").status_code == 404

        # The real streaming path: an SSE turn streams to a logged-in operator.
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "hi"},
            headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
        ) as stream:
            assert stream.status_code == 200
            assert "text/event-stream" in stream.headers["content-type"]
            body = "".join(stream.iter_text())
        assert "data:" in body

        # The Telegram callback drives a real turn with no cookie and no CSRF
        # token anywhere in sight - its auth is the chat-id allowlist, and it must
        # not have been coupled to the HTTP session model. Driven on the app's OWN
        # loop (the turn runs under the supervisor that lives there).
        portal = client.portal
        assert portal is not None, "TestClient must be entered to expose its portal"
        events = portal.call(_collect, _FakeBot.instances[0].on_message("hi"))
        assert events, "the Telegram bridge produced no turn events under auth"


# --- the browser surface ----------------------------------------------------


def _built_dist(tmp_path: Path) -> Path:
    """A minimal stand-in for the built frontend bundle."""
    dist = tmp_path / "dist"
    (dist / "login").mkdir(parents=True)
    (dist / "index.html").write_text("<html>dashboard</html>", encoding="utf-8")
    (dist / "login" / "index.html").write_text("<html>login</html>", encoding="utf-8")
    (dist / "login.js").write_text("// login bundle", encoding="utf-8")
    (dist / "agent.js").write_text("// dashboard bundle", encoding="utf-8")
    return dist


def test_unauthenticated_navigation_redirects_to_the_login_page(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A browser hitting the dashboard gets the login page, not a bare 401 - and
    the dashboard's own assets stay gated."""
    settings = _settings(tmp_path, web_dist=_built_dist(tmp_path))
    client = TestClient(
        create_app(collector=fake_collector, settings=settings), follow_redirects=False
    )

    nav = client.get("/", headers={"Accept": "text/html"})
    assert nav.status_code == 303
    assert nav.headers["location"].startswith("/login/")

    # The login page and its bundle are reachable...
    assert client.get("/login/").status_code == 200
    assert client.get("/login.js").status_code == 200
    # ...the dashboard bundle is not.
    assert client.get("/agent.js").status_code == 401

    # An API call still gets a JSON 401 rather than a redirect: the frontend
    # needs a status to react to, not an HTML login page in its fetch.
    api = client.get("/api/stats", headers={"Accept": "application/json"})
    assert api.status_code == 401


def test_login_redirect_target_stays_local(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The ``next`` parameter must not become an open redirect."""
    settings = _settings(tmp_path, web_dist=_built_dist(tmp_path))
    client = TestClient(
        create_app(collector=fake_collector, settings=settings), follow_redirects=False
    )
    nav = client.get("/agents/", headers={"Accept": "text/html"})
    assert nav.status_code == 303
    location = nav.headers["location"]
    assert location.startswith("/login/")
    assert "evil" not in location
    assert "//" not in location.removeprefix("/login/")


# --- review round 1 regressions ---------------------------------------------


def test_non_ascii_credentials_are_refused_not_crashed(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A garbage credential header must be REFUSED, never raise.

    Starlette decodes headers as latin-1, so a raw byte above 0x7F arrives as a
    non-ASCII str - and `hmac.compare_digest` raises TypeError on those. That
    turned an unauthenticated request into a 500 plus a traceback in the journal
    (review round 1, finding 1).

    Driven over a REAL socket: httpx/TestClient refuse to SEND such a header, so
    a TestClient-only version of this test would pass while production 500s.
    """
    import uvicorn

    port = _free_port()
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, port=port))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started, "uvicorn did not start"

        def raw_request(headers: bytes) -> int:
            sock = socket.create_connection(("127.0.0.1", port), timeout=10)
            try:
                sock.sendall(
                    b"GET /api/stats HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                    + headers
                    + b"Connection: close\r\n\r\n"
                )
                status = sock.recv(64).split(b" ")[1]
            finally:
                sock.close()
            return int(status)

        # A non-ASCII bearer token, and a non-ASCII CSRF header.
        assert raw_request(b"Authorization: Bearer \xff\xfe\r\n") == 401
        assert raw_request(b"X-Scufris-CSRF: \xff\xfe\r\n") == 401
        # ...and the plain unauthenticated case still behaves.
        assert raw_request(b"") == 401
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_token_matches_is_total_over_any_string() -> None:
    """The comparison helper must never raise, whatever bytes reach it."""
    from scufris.auth import token_matches

    for presented in ("\xff\xfe", "caf\xe9", "\ud800", "ok"):
        assert token_matches(presented, "expected") is False
    assert token_matches("same", "same") is True
    assert token_matches("\xff", "\xff") is True


def test_agent_cli_env_does_not_carry_the_machine_token(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent CLI's environment must NOT hold the dashboard's API credential.

    Everything the model runs inherits that environment - every shell command,
    every sub-agent, and the den MCP server whatever its permission mode. Asserts
    against the env actually handed to the subprocess, not the declared MCP env
    dict: the earlier version of this check inspected the dict and was therefore
    vacuous while `_codex_env` leaked the token through `os.environ` (review round
    1, finding 2).
    """
    from scufris.agent import _codex_env

    settings = _settings(tmp_path)
    app = create_app(collector=fake_collector, settings=settings)
    token = app.state.api_token

    # The mint must not have gone through the environment at all...
    assert os.environ.get("SCUFRIS_API_TOKEN") != token
    # ...and even if something else set it, the CLI env is stripped.
    monkeypatch.setenv("SCUFRIS_API_TOKEN", token)
    assert "SCUFRIS_API_TOKEN" not in _codex_env(settings)


def test_two_apps_do_not_clobber_each_others_machine_token(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """Each app carries its own token, so creating a second one does not lock the
    first one's tools out (review round 1, finding 3)."""
    first_settings = _settings(tmp_path / "a")
    second_settings = _settings(tmp_path / "b")
    first = create_app(collector=fake_collector, settings=first_settings)
    second = create_app(collector=fake_collector, settings=second_settings)

    assert first.state.api_token != second.state.api_token
    assert first_settings.auth_api_token == first.state.api_token
    assert second_settings.auth_api_token == second.state.api_token

    # Each app still accepts its OWN token and refuses the other's.
    for app, own, other in (
        (first, first.state.api_token, second.state.api_token),
        (second, second.state.api_token, first.state.api_token),
    ):
        client = TestClient(app)
        assert (
            client.get(
                "/api/stats", headers={"Authorization": f"Bearer {own}"}
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/api/stats", headers={"Authorization": f"Bearer {other}"}
            ).status_code
            == 401
        )


def test_cross_origin_login_is_refused_before_it_can_burn_the_lockout(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A page the operator visits must not be able to lock them out of their own
    dashboard by firing cross-origin logins (review round 1, finding 5)."""
    settings = _settings(tmp_path, auth_login_max_failures=3)
    client = TestClient(create_app(collector=fake_collector, settings=settings))

    for _ in range(10):
        resp = client.post(
            "/api/auth/login",
            json={"password": "guess"},
            headers={"Origin": "http://evil.example"},
        )
        assert resp.status_code == 403, resp.text

    # The real operator can still log in: those attempts never counted.
    assert _login(client)


def test_throttle_bounds_a_distributed_attempt_and_does_not_grow_forever() -> None:
    """Per-source alone is evaded by an IPv6 /64 and leaks memory doing it, so
    there is a global ceiling and aged entries are pruned (round 1, finding 4)."""
    throttle = LoginThrottle(max_failures=3, window_seconds=60.0)
    now = 1000.0

    # Each address gets its own allowance, but the GLOBAL ceiling (max * 5) stops
    # the sweep well before the address space is exhausted.
    blocked_at = None
    for i in range(100):
        source = f"2001:db8::{i}"
        if not throttle.allowed(source, now=now):
            blocked_at = i
            break
        throttle.record_failure(source, now=now)
    assert blocked_at is not None, "a distributed attempt was never bounded"
    assert blocked_at <= 15, f"global ceiling let {blocked_at} sources through"

    # A locked-out caller is told when to come back.
    assert throttle.retry_after("2001:db8::0", now=now) > 0

    # Once the window passes, everything is released AND the memory is reclaimed.
    later = now + 61.0
    assert throttle.allowed("2001:db8::0", now=later)
    throttle.record_failure("fresh", now=later)
    assert len(throttle._failures) == 1, "aged-out sources were not pruned"  # noqa: SLF001


def test_expired_sessions_are_swept_not_kept_until_presented(tmp_path: Path) -> None:
    """An abandoned session must not sit in the store forever (finding 6)."""
    path = tmp_path / "auth_sessions.json"
    store = SessionStore(path)
    old = [store.create(now=1000.0) for _ in range(5)]
    live = store.create(now=5000.0)

    removed = store.prune(now=5000.0, idle=100.0, absolute=1000.0)
    assert removed == 5
    for session in old:
        assert store.get(session.id, now=5000.0, idle=100.0, absolute=1000.0) is None
    assert store.get(live.id, now=5000.0, idle=100.0, absolute=1000.0) is not None
    assert "sessions" in path.read_text(encoding="utf-8")


def test_startup_sweeps_sessions_that_expired_while_down(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A restart clears out what expired during the downtime."""
    store = SessionStore(tmp_path / "auth_sessions.json")
    stale = store.create(now=1.0)

    settings = _settings(tmp_path, auth_session_idle_seconds=1.0)
    app = create_app(collector=fake_collector, settings=settings)
    assert (
        app.state.sessions.get(stale.id, now=time.time(), idle=1.0, absolute=1.0)
        is None
    )
