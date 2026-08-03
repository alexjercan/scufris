"""What a request must carry to change state, and the surfaces that enforce it.

The load-bearing test is ``test_authenticated_session_and_csrf_boundary``: it
derives the protected surface from ``iter_routes(app)`` rather than from a
hand-written list, so a route added later is covered by existing rather than by
someone remembering to extend a list.

Covers the session cookie, the CSRF token, the Origin check, the login throttle,
the browser redirects an unauthenticated navigation gets, and the two bridges
that carry a session onward - the event stream and Telegram.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Route
from test_auth import ORIGIN, PASSWORD, _login, _settings

from scufris.api.routes import iter_routes
from scufris.app import create_app
from scufris.auth import CSRF_HEADER, SESSION_COOKIE, LoginThrottle
from scufris.enums import Backend
from scufris_host import Collector


async def _collect(events: Any) -> list[Any]:
    """Drain an async event stream into a list."""
    return [event async for event in events]


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

    The surface is ENUMERATED from ``iter_routes(app)`` so a new route is
    covered without touching this test - including one that lives on an included
    router, which `app.routes` alone no longer walks into."""
    app = create_app(collector=fake_collector, settings=_settings(tmp_path))
    from scufris.auth import PUBLIC_PATHS, PUBLIC_STATIC_PATHS

    client = TestClient(app)

    checked_unauth = 0
    checked_csrf = 0
    for route in iter_routes(app):
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
    for route in iter_routes(app):
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

    # Guard the guard: set just under today's real coverage (81 and 31), not at
    # a token floor. A floor of 40 stayed satisfied with the entire host, auth
    # and config surface silently dropped from the sweep, which is precisely the
    # failure `iter_routes` exists to make impossible.
    assert checked_unauth > 75, f"sweep covered only {checked_unauth} route/methods"
    assert checked_csrf > 28, f"CSRF sweep covered only {checked_csrf} route/methods"


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


def test_a_forged_session_id_is_refused(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_settings(tmp_path))
    )
    client.cookies.set(SESSION_COOKIE, "0" * 43)
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
    monkeypatch.setattr("scufris.telegram.wiring.TelegramBot", _FakeBot)
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
    (dist / "host").mkdir(parents=True)
    (dist / "index.html").write_text("<html>dashboard</html>", encoding="utf-8")
    (dist / "login" / "index.html").write_text("<html>login</html>", encoding="utf-8")
    (dist / "login.js").write_text("// login bundle", encoding="utf-8")
    (dist / "agent.js").write_text("// dashboard bundle", encoding="utf-8")
    (dist / "host" / "index.html").write_text("<html>host</html>", encoding="utf-8")
    (dist / "host.js").write_text("// host bundle", encoding="utf-8")
    return dist


def test_the_host_page_requires_a_session(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The host action page is protected by DEFAULT, with no allowlist entry.

    It is served as a static page (no FastAPI route of its own), so its protection
    comes entirely from the deny-by-default middleware and from `host` NOT being in
    `PUBLIC_STATIC_PATHS`. That is worth pinning rather than assuming: this is the
    page that approves root commands, and the login page is the only static thing
    on this server that is reachable without a session.
    """
    settings = _settings(tmp_path, web_dist=_built_dist(tmp_path))
    app = create_app(collector=fake_collector, settings=settings)
    client = TestClient(app, follow_redirects=False)

    # A browser is sent to the login page; the bundle is refused outright.
    nav = client.get("/host/", headers={"Accept": "text/html"})
    assert nav.status_code == 303
    assert nav.headers["location"].startswith("/login/")
    assert client.get("/host.js").status_code == 401
    # And the API the page lives on answers with a status the frontend can react to.
    assert client.get("/api/host/actions").status_code == 401

    # The page is not in the public allowlist at all.
    from scufris.auth import PUBLIC_STATIC_PATHS

    assert not [path for path in PUBLIC_STATIC_PATHS if "host" in path]

    # With a session it is served.
    login = client.post(
        "/api/auth/login", json={"password": PASSWORD}, headers={"Origin": ORIGIN}
    )
    assert login.status_code == 200, login.text
    assert client.get("/host/").status_code == 200


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
