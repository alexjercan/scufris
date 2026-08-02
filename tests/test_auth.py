"""The credential itself: hashing it, the policy that may waive it, and the session it mints.

Covers the password hash, the loopback fail-closed policy (an app that binds
beyond loopback without credentials refuses to start), the login/logout round
trip and the session id rotation it performs, session lifetime - both idle
expiry and the absolute cap - and the store those sessions survive a restart in.

The helpers here (``PASSWORD``, ``ORIGIN``, ``_settings``, ``_login``,
``_free_port``) are auth-domain-local and imported by
``tests/test_auth_boundary.py`` and ``tests/test_auth_machine.py``.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from scufris.app import create_app
from scufris.auth import (
    CSRF_COOKIE,
    CSRF_HEADER,
    SESSION_COOKIE,
    AuthConfigError,
    SessionStore,
    hash_password,
    verify_password,
)
from scufris.config import Settings
from scufris.db import Database, state_database
from scufris.enums import AuthPolicy
from scufris.metrics import Collector

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
    monkeypatch.setattr("scufris.auth.store.time.time", lambda: now + 5.0)
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
        monkeypatch.setattr("scufris.auth.store.time.time", lambda o=offset: now + o)
        assert client.get("/api/stats").status_code == 200
    monkeypatch.setattr("scufris.auth.store.time.time", lambda: now + 11.0)
    assert client.get("/api/stats").status_code == 401


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


# Where the "a live session id is not world-readable" proof went: sessions are
# rows in the state database now, so the file whose mode matters is `scufris.db`
# and its -wal/-shm siblings. That is
# `test_db_state_boundary.py::test_state_database_is_private_with_a_live_session`,
# which logs in through the real app rather than constructing a store.


def test_revoke_all_invalidates_every_session(database: Database) -> None:
    store = SessionStore(database)
    a = store.create(now=1000.0)
    b = store.create(now=1000.0)
    store.revoke_all()
    assert store.get(a.id, now=1000.0, idle=100.0, absolute=100.0) is None
    assert store.get(b.id, now=1000.0, idle=100.0, absolute=100.0) is None


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


def test_expired_sessions_are_swept_not_kept_until_presented(
    database: Database,
) -> None:
    """An abandoned session must not sit in the store forever (finding 6)."""
    store = SessionStore(database)
    old = [store.create(now=1000.0) for _ in range(5)]
    live = store.create(now=5000.0)

    removed = store.prune(now=5000.0, idle=100.0, absolute=1000.0)
    assert removed == 5
    for session in old:
        assert store.get(session.id, now=5000.0, idle=100.0, absolute=1000.0) is None
    assert store.get(live.id, now=5000.0, idle=100.0, absolute=1000.0) is not None


def test_startup_sweeps_sessions_that_expired_while_down(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A restart clears out what expired during the downtime."""
    # The process-wide accessor, so this store and the app below share ONE
    # database - which is the whole point of the sweep being at startup.
    store = SessionStore(state_database(tmp_path))
    stale = store.create(now=1.0)

    settings = _settings(tmp_path, auth_session_idle_seconds=1.0)
    app = create_app(collector=fake_collector, settings=settings)
    assert (
        app.state.sessions.get(stale.id, now=time.time(), idle=1.0, absolute=1.0)
        is None
    )
