"""Every question the enforcement point asks, answered in one module.

The enforcement point itself lives in ``app.py`` as ONE middleware, deny by
default. This module is what it consults: which paths are public, which methods
are unsafe, which paths a machine may never reach, whether authentication is
required at all, and whether a state-changing request came from this dashboard.

``PUBLIC_PATHS`` and ``PUBLIC_STATIC_PATHS`` are defined here and nowhere else.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from ..config import Settings
from ..enums import AuthPolicy

# Cookie and header names. The session cookie is HttpOnly (JavaScript must never
# read it); the CSRF cookie deliberately is NOT, because the frontend has to echo
# it back in a header - that asymmetry IS the double-submit check.
SESSION_COOKIE = "scufris_session"
CSRF_COOKIE = "scufris_csrf"
CSRF_HEADER = "X-Scufris-CSRF"

# The machine credential the app's own MCP tool subprocesses present. Minted per
# process, never persisted, so there is no credential at rest.
API_TOKEN_ENV = "SCUFRIS_API_TOKEN"

# API paths reachable without a session. Deliberately tiny: the login endpoint
# and the "am I logged in" probe the login page needs to avoid a redirect loop.
# Everything else is denied by default, and the DoD sweep in tests/test_auth.py
# enumerates app.routes to prove it.
PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/api/auth/login",
        "/api/auth/session",
    }
)

# Static assets the login page itself needs, plus the page. Without these the
# operator cannot reach a login form to stop being unauthenticated.
PUBLIC_STATIC_PATHS: frozenset[str] = frozenset(
    {
        "/login",
        "/login/",
        "/login/index.html",
        "/login.js",
        "/favicon.ico",
    }
)

# Methods that change state, and so need the CSRF token and an origin check.
UNSAFE_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Paths a MACHINE may never reach, whatever credential it holds.
#
# The middleware accepts two identities, and for almost every route that is
# right: the app's own MCP tool subprocesses present the per-process bearer
# token and get on with their work. Approving a privileged host action is the
# exception, and it is not a small one. The bearer branch short-circuits BEFORE
# the session lookup and before the CSRF and origin checks, and the agent CLI
# subprocesses hold exactly that token - so without this list an agent could
# approve its own proposal, and the whole propose/preview/approve contract would
# be a description of something the code does not do.
#
# An approval is an OPERATOR act. It requires a session, and therefore a human
# who logged in.
#
# Matched by regex because the action id is in the path. The verb alternation is
# EXPLICIT, so a route added under this prefix is NOT covered automatically -
# `test_every_mutating_host_route_is_operator_only` enumerates `app.routes` and
# fails when a mutating host route is missing from this pattern, which is what
# actually keeps the two in step.
OPERATOR_ONLY_PATTERN = re.compile(
    r"^/api/host/actions/[^/]+/(approve|deny|revert|cancel)/?$"
    # Running the scheduled checks NOW is the operator's button too. It reads the
    # host rather than changing it, but it can escalate a breach into a proposal and
    # it makes the machine do work on demand - and no agent has any use for it, so it
    # stays on the operator's side of the line rather than becoming the first
    # exception to "every mutating host route is operator-only".
    r"|^/api/host/digests/run/?$"
)

# Hostnames/addresses that mean "this machine only". A bind to any of these is
# not reachable from the network, which is what makes open development mode
# defensible.
_LOOPBACK_HOSTS: frozenset[str] = frozenset(
    {"127.0.0.1", "localhost", "::1", "[::1]", "0:0:0:0:0:0:0:1"}
)


class AuthConfigError(RuntimeError):
    """The authentication configuration cannot serve safely.

    Raised at app construction, never at request time: a misconfigured deployment
    must fail to start rather than start and quietly serve open.
    """


def operator_only(path: str) -> bool:
    """Whether ``path`` demands a real operator session, not a machine token."""
    return OPERATOR_ONLY_PATTERN.match(path) is not None


def is_loopback_host(host: str) -> bool:
    """Whether a bind address reaches only this machine."""
    return host.strip().lower() in _LOOPBACK_HOSTS


def auth_required(settings: Settings) -> bool:
    """Whether this configuration must authenticate requests.

    ``auto`` (the default) resolves from the BIND ADDRESS: open on loopback,
    mandatory anywhere else. That is what keeps pytest, the examples and the mock
    backend free of a login dance while making the deployed LAN bind protected
    without the operator opting in.
    """
    if settings.auth_mode is AuthPolicy.REQUIRED:
        return True
    if settings.auth_mode is AuthPolicy.DISABLED:
        return False
    return not is_loopback_host(settings.host)


def validate_auth_config(settings: Settings) -> None:
    """Refuse a configuration that would serve a network bind without a gate.

    Called from ``create_app``, so the failure is "the service does not start",
    which is the whole point of a fail-closed gate. A warn-and-serve here would
    reopen exactly the hole this package exists to close.
    """
    loopback = is_loopback_host(settings.host)
    if settings.auth_mode is AuthPolicy.DISABLED and not loopback:
        raise AuthConfigError(
            f"SCUFRIS_AUTH_MODE=disabled is refused on a non-loopback bind "
            f"(host={settings.host!r}). Authentication cannot be turned off for a "
            "network-reachable dashboard; bind 127.0.0.1 for open development."
        )
    if auth_required(settings) and not settings.auth_password_hash:
        raise AuthConfigError(
            f"authentication is required (host={settings.host!r}, "
            f"mode={settings.auth_mode.value}) but no credential is configured. "
            "Set SCUFRIS_AUTH_PASSWORD_HASH - generate it with `scufris "
            "hash-password`. Refusing to serve an unauthenticated dashboard on a "
            "network-reachable address."
        )
    if settings.hostd_secret and not settings.auth_password_hash:
        # Host agency requires an authenticated operator, whatever the bind
        # address. Approving a privileged action is a HUMAN act, and with no
        # credential configured there is no human to be - every approval
        # endpoint would accept an anonymous caller, which on loopback means
        # any process on this machine, which includes the shell the model runs
        # its own commands in.
        #
        # The bind-address rule above is about the NETWORK. This one is not: it
        # is about there being someone to attribute an approval to.
        raise AuthConfigError(
            "SCUFRIS_HOSTD_SECRET is set (the privileged host helper is "
            "enabled) but no operator credential is configured. Approving a "
            "host action is a human act and there would be no human to be. Set "
            "SCUFRIS_AUTH_PASSWORD_HASH - generate it with `scufris "
            "hash-password` - or unset SCUFRIS_HOSTD_SECRET to run without host "
            "agency."
        )


def same_origin(
    origin_header: str | None, referer: str | None, host: str | None
) -> bool:
    """Whether a state-changing request came from this dashboard's own origin.

    A browser sends ``Origin`` on every state-changing request; some privacy
    configurations strip it, so ``Referer`` is accepted as the fallback. Neither
    present is REFUSED rather than trusted - a cookie-authenticated request with
    no provenance at all is not a shape worth accepting, and every legitimate
    caller either is a browser (sends one) or uses the bearer token (skips this
    check entirely).
    """
    if not host:
        return False
    candidate = origin_header or referer
    if not candidate:
        return False
    parsed = urlsplit(candidate)
    if not parsed.netloc:
        return False
    return parsed.netloc.lower() == host.lower()


def safe_next_path(raw: str | None) -> str:
    """Sanitize a post-login redirect target down to a local path.

    Anything protocol-relative (``//evil.example``), absolute, or backslashed is
    discarded rather than repaired: this value ends up in a ``Location`` header,
    and an open redirect on the login page is a phishing primitive.
    """
    if not raw or not raw.startswith("/"):
        return "/"
    if raw.startswith("//") or raw.startswith("/\\"):
        return "/"
    if "\\" in raw or "\n" in raw or "\r" in raw:
        return "/"
    return raw


def session_cookie_kwargs(
    *, secure: bool, max_age: int, http_only: bool = True
) -> dict[str, Any]:
    """Cookie attributes shared by the session and CSRF cookies.

    ``secure`` is derived from the request scheme rather than hardcoded: the
    supported LAN deployment is plaintext HTTP, and a hardcoded ``Secure`` would
    make the cookie silently never stick there. A TLS-terminated deployment gets
    it automatically.
    """
    return {
        "httponly": http_only,
        "samesite": "lax",
        "secure": secure,
        "path": "/",
        "max_age": max_age,
    }
