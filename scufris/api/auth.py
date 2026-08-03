"""The operator session: the gate, the enforcement middleware, and the routes.

Three pieces, in the order a request meets them:

- ``SessionGate`` - the ONE place that reads, mints and revokes a session, and
  the only thing that answers "who is asking". Every identity question in the
  app goes through it, so the audit's answer, the middleware's answer and the
  agent-chat guard's answer cannot drift apart;
- ``auth_middleware`` - the single enforcement point, deny by default;
- ``build_auth_router`` - ``/api/auth/login|logout|session``.

``scufris.auth`` (the package) holds the primitives: password hashing, the
session store, the public-path allowlist, the CSRF and origin checks. This
module is the HTTP surface over them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable
from urllib.parse import quote

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from ..auth import (
    CSRF_COOKIE,
    CSRF_HEADER,
    PUBLIC_PATHS,
    PUBLIC_STATIC_PATHS,
    SESSION_COOKIE,
    UNSAFE_METHODS,
    LoginThrottle,
    Session,
    SessionStore,
    auth_required,
    bearer_token,
    operator_only,
    safe_next_path,
    same_origin,
    session_cookie_kwargs,
    token_matches,
    verify_password,
)
from ..auth import (
    now as auth_now,
)
from ..config import Settings
from ..hostd.audit import Requester

logger = logging.getLogger(__name__)

#: A Starlette ``BaseHTTPMiddleware`` dispatch: the request, and the rest of the
#: stack to call (or not).
Dispatch = Callable[
    [Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]
]


class LoginRequest(BaseModel):
    """The login body. Carries the password only - there is one operator, so
    there is no username to get wrong."""

    password: str


class AuthSession(BaseModel):
    """The authentication posture of the caller and of this deployment.

    `required` lets the frontend skip the login flow entirely in loopback
    development instead of guessing from a status code."""

    authenticated: bool
    required: bool


def deny(request: Request, status: int, detail: str) -> Response:
    """Refuse a request the way its caller can actually use.

    A browser NAVIGATION gets the login page (a bare 401 would show a blank
    screen); an API call gets a JSON status the frontend can react to. The
    redirect target is sanitized - it lands in a Location header, and an open
    redirect on a login page is a phishing primitive.

    A free function: refusing reads the request and nothing else, so it is not
    the session gate's business.
    """
    wants_html = "text/html" in request.headers.get("accept", "")
    if status == 401 and request.method == "GET" and wants_html:
        target = quote(safe_next_path(request.url.path), safe="/")
        return RedirectResponse(f"/login/?next={target}", status_code=303)
    return JSONResponse({"detail": detail}, status_code=status)


class SessionGate:
    """Reads, mints and revokes the operator session, and names the caller.

    Every "who is asking" answer in the app comes from here, and every one of
    them is derived from the CREDENTIAL rather than from the request body - the
    one question a caller must not be able to answer about itself. The gate is
    shared rather than private to the auth router: the middleware enforces with
    it, the host routes stamp the audit with it, and `/api/agents/{id}/chat`
    asks it whether the caller is an agent.

    Every store call is offloaded: ``SessionStore`` opens a transaction, which
    holds SQLite's single write lock, and the engine refuses to open one on a
    thread with a running event loop (``scufris/db/engine.py``). Doing it here
    once is why no caller has to remember to.
    """

    def __init__(self, settings: Settings, sessions: SessionStore) -> None:
        self.settings = settings
        self._sessions = sessions
        #: Whether a credential is demanded at all, decided once at construction
        #: from the bind address and the configured policy (``auth_required``).
        #: Published as ``app.state.auth_required`` and reported by
        #: ``/api/auth/session``.
        self.required = auth_required(settings)

    async def session_of(self, request: Request) -> Session | None:
        """The live session this request carries, renewed, or None."""
        return await asyncio.to_thread(
            self._sessions.get,
            request.cookies.get(SESSION_COOKIE),
            now=auth_now(),
            idle=self.settings.auth_session_idle_seconds,
            absolute=self.settings.auth_session_max_seconds,
        )

    async def issue(self, response: Response, request: Request) -> None:
        """Mint a session and attach its cookies to ``response``.

        The session id ROTATES on every login (the caller revokes the old one
        first), which is what closes session fixation: an id an attacker planted
        in the browser before login is never the id that ends up authenticated.
        """
        session = await asyncio.to_thread(self._sessions.create, now=auth_now())
        secure = request.url.scheme == "https"
        max_age = int(self.settings.auth_session_max_seconds)
        response.set_cookie(
            SESSION_COOKIE,
            session.id,
            **session_cookie_kwargs(secure=secure, max_age=max_age),
        )
        # Readable by JavaScript ON PURPOSE: the frontend echoes it back in the
        # CSRF header, and a cross-site attacker can send the cookie but cannot
        # read it to build the header.
        response.set_cookie(
            CSRF_COOKIE,
            session.csrf,
            **session_cookie_kwargs(secure=secure, max_age=max_age, http_only=False),
        )

    async def revoke(self, request: Request) -> None:
        """Revoke whatever session this request carries, server-side."""
        await asyncio.to_thread(
            self._sessions.revoke, request.cookies.get(SESSION_COOKIE)
        )

    async def prune(self, now: float) -> None:
        """Drop records nobody will ever present again.

        Called on login - the one moment a new record is added, so the one
        moment the store is worth sweeping. ``now`` is passed in rather than read
        here so the sweep and the throttle decision that admitted the login share
        one instant.
        """
        await asyncio.to_thread(
            self._sessions.prune,
            now=now,
            idle=self.settings.auth_session_idle_seconds,
            absolute=self.settings.auth_session_max_seconds,
        )

    async def caller_is_agent(self, request: Request) -> bool:
        """Whether this caller is one of the app's own tool subprocesses (an AGENT)
        rather than the operator.

        Derived from the CREDENTIAL, never from the body - the same rule
        ``requester_identity`` follows and for the same reason: "who is asking" is
        exactly the question a caller must not be able to answer about itself. A
        session is the operator; a bearer token is a machine, which is to say an
        agent; neither (only reachable with auth off) is nobody, and nobody is not
        an agent.
        """
        session = await self.session_of(request)
        if session is not None:
            return False
        return bearer_token(request.headers.get("authorization")) is not None

    async def operator_identity(self, request: Request) -> str:
        """Who approved, for the record. One operator, so this is traceability."""
        session = await self.session_of(request)
        return f"operator:{session.id[:8]}" if session is not None else "operator"

    async def requester_identity(
        self, request: Request, *, agent: str = "", run: str = ""
    ) -> Requester:
        """Who asked, derived from the CREDENTIAL rather than from the body.

        "Who asked" is the one question the audit exists to answer, so it must
        not be answerable by the caller. The first version read
        `actor = "agent" if body.agent else ...`, and the MCP tool sent no
        `agent` field - so every agent-originated proposal was written into the
        root-owned log as having been asked for by the operator (review round 1,
        R1.6). A body field is a hint about WHICH agent; the credential is the
        fact about what kind of caller it is.
        """
        session = await self.session_of(request)
        if session is not None:
            return Requester(actor=f"operator:{session.id[:8]}", agent=agent, run=run)
        if bearer_token(request.headers.get("authorization")) is not None:
            # A machine credential: this app's own tool subprocess, which is to
            # say an agent. It may name itself, but it cannot claim to be human.
            return Requester(actor="agent", agent=agent or "orchestrator", run=run)
        # Neither: only reachable with auth off, where the caller is anonymous
        # and the record should say exactly that rather than guess.
        return Requester(actor="unauthenticated", agent=agent, run=run)


def auth_middleware(gate: SessionGate, api_token: str) -> Dispatch:
    """Build the single enforcement point: deny by default, allow by exception.

    Registered BEFORE the request logger so the logger stays outermost and a
    denial is still logged. Every route is gated unless its path is in the small
    public allowlist, so a route added tomorrow is protected by existing -
    `tests/test_auth_boundary.py` sweeps `iter_routes(app)` to prove it.

    Two identities are accepted. A browser presents the session cookie and is
    subject to the CSRF and origin checks (it carries ambient credentials that
    another site could try to ride). A machine caller - this app's own MCP tool
    subprocesses - presents the per-process bearer token and is not: it has no
    cookie to ride, and requiring a CSRF token would break every tool.
    """

    async def enforce_auth(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        # Operator-only paths are decided BEFORE the bearer branch AND before
        # the auth_on short-circuit, not inside either. Approving a privileged
        # host action is a human act; the machine token belongs to the app's own
        # tool subprocesses, which is to say to the agent. Deciding it later
        # would leave the framework's central claim untrue - and deciding it
        # only when auth is on would mean a loopback deployment lets an agent
        # approve its own proposal, which has nothing to do with the bind
        # address (see auth.OPERATOR_ONLY_PATTERN).
        if operator_only(path):
            # An operator-only path needs a real SESSION, and the check does not
            # look at the credential presented - it looks at whether one that
            # identifies a human was. The first version of this asked "is a
            # bearer token present?", which meant a caller that sent NO header at
            # all sailed through to the `auth_on` short-circuit below and
            # executed a root command anonymously. On loopback that is any
            # process on this machine, including the shell the model runs its own
            # commands in (`curl -XPOST .../approve`). Review round 1, R1.1.
            #
            # `validate_auth_config` refuses to build an app with host agency and
            # no operator credential, so on a correct deployment this branch is
            # about WHICH credential. It is written to stand alone anyway: a
            # guarantee that depends on a check somewhere else holding is not a
            # guarantee.
            session = await gate.session_of(request)
            if session is None:
                return deny(
                    request,
                    403 if bearer_token(request.headers.get("authorization")) else 401,
                    "approving a host action needs an operator session; a machine "
                    "credential cannot do it and neither can an anonymous caller",
                )
            # Fully self-contained, including CSRF and origin - deliberately not
            # falling through to the generic block below, because that block is
            # skipped when auth is off and these paths must not be.
            if request.method in UNSAFE_METHODS:
                if not same_origin(
                    request.headers.get("origin"),
                    request.headers.get("referer"),
                    request.headers.get("host"),
                ):
                    return deny(request, 403, "cross-origin request refused")
                if not token_matches(request.headers.get(CSRF_HEADER), session.csrf):
                    return deny(request, 403, "missing or invalid CSRF token")
            return await call_next(request)
        if not gate.required:
            return await call_next(request)
        if path in PUBLIC_PATHS or path in PUBLIC_STATIC_PATHS:
            return await call_next(request)

        presented = bearer_token(request.headers.get("authorization"))
        if presented is not None:
            # No operator-only check here: the block above returned on every one
            # of those paths, whatever the credential and whatever the bind
            # address. There is ONE enforcement point, and a reader only has to
            # trust that one (review round 2, R2.6 removed the dead second).
            if token_matches(presented, api_token):
                return await call_next(request)
            return deny(request, 401, "invalid credentials")

        session = await gate.session_of(request)
        if session is None:
            return deny(request, 401, "authentication required")
        if request.method in UNSAFE_METHODS:
            if not same_origin(
                request.headers.get("origin"),
                request.headers.get("referer"),
                request.headers.get("host"),
            ):
                return deny(request, 403, "cross-origin request refused")
            if not token_matches(request.headers.get(CSRF_HEADER), session.csrf):
                return deny(request, 403, "missing or invalid CSRF token")
        return await call_next(request)

    return enforce_auth


def build_auth_router(gate: SessionGate, throttle: LoginThrottle) -> APIRouter:
    """The operator session endpoints, over an explicit gate and throttle."""
    router = APIRouter()

    @router.post("/api/auth/login")
    async def post_auth_login(request: Request, body: LoginRequest) -> Response:
        """Exchange the operator password for a session.

        Public (it has to be), throttled per source, and deliberately uniform in
        its failure: a wrong password and an unconfigured credential answer the
        same way, so this endpoint cannot be used to probe the deployment.

        Origin-checked despite being public. Without it, any page the operator
        happens to visit can fire cross-origin logins at the dashboard's LAN
        address until the lockout window burns, denying the REAL operator their
        own login. The login page is same-origin, so nothing legitimate is
        affected - and the check runs BEFORE the throttle, so a refused
        cross-origin attempt cannot count toward the lockout it was trying to
        trigger. Review round 1, finding 5.
        """
        if not same_origin(
            request.headers.get("origin"),
            request.headers.get("referer"),
            request.headers.get("host"),
        ):
            return JSONResponse(
                {"detail": "cross-origin request refused"}, status_code=403
            )
        source = request.client.host if request.client else "unknown"
        moment = auth_now()
        if not throttle.allowed(source, now=moment):
            return JSONResponse(
                {"detail": "too many failed attempts; try again later"},
                status_code=429,
                headers={"Retry-After": str(throttle.retry_after(source, now=moment))},
            )
        stored = gate.settings.auth_password_hash
        if not stored or not verify_password(body.password, stored):
            throttle.record_failure(source, now=moment)
            logger.warning("auth: failed login from %s", source)
            return JSONResponse({"detail": "invalid credentials"}, status_code=401)
        throttle.record_success(source)
        # Rotate: whatever session the browser was carrying is revoked, not reused.
        await gate.revoke(request)
        await gate.prune(moment)
        response = JSONResponse({"authenticated": True})
        await gate.issue(response, request)
        logger.info("auth: operator logged in from %s", source)
        return response

    @router.post("/api/auth/logout")
    async def post_auth_logout(request: Request) -> Response:
        """Revoke this session server-side and clear its cookies."""
        await gate.revoke(request)
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(SESSION_COOKIE, path="/")
        response.delete_cookie(CSRF_COOKIE, path="/")
        return response

    @router.get("/api/auth/session")
    async def get_auth_session(request: Request) -> AuthSession:
        """Whether this caller has a session, and whether one is needed at all.

        Public so the login page can ask without tripping a redirect loop. It
        reports posture only - never who, never the token."""
        if not gate.required:
            return AuthSession(authenticated=True, required=False)
        session = await gate.session_of(request)
        return AuthSession(authenticated=session is not None, required=True)

    return router
