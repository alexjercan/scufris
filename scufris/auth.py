"""Single-operator authentication for the dashboard.

Scufris is deployed to the LAN with no HTTP authentication at all, which the
host-operator epic cannot build on. This module is the whole mechanism:

- a password verified against a stdlib ``scrypt`` hash delivered through the
  same sops dotenv the Telegram token already rides,
- an opaque session id in an ``HttpOnly`` cookie, backed by a revocable
  server-side record under ``state_dir``,
- a per-session CSRF token for the double-submit check,
- failed-login throttling.

The enforcement point itself lives in ``app.py`` as ONE middleware, deny by
default; this module owns the primitives and the policy questions (is
authentication required at all, which paths are public). See
``tasks/20260729-125015/DECISION.md`` for why each of these is what it is - in
particular why the session is server-side rather than a signed cookie, and why
loopback is not implicitly trusted.

Nothing here logs a password, a session id, or the machine token.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

from .config import Settings
from .enums import AuthPolicy

logger = logging.getLogger(__name__)

# Cookie and header names. The session cookie is HttpOnly (JavaScript must never
# read it); the CSRF cookie deliberately is NOT, because the frontend has to echo
# it back in a header - that asymmetry IS the double-submit check.
SESSION_COOKIE = "scufris_session"
CSRF_COOKIE = "scufris_csrf"
CSRF_HEADER = "X-Scufris-CSRF"

# The machine credential the app's own MCP tool subprocesses present. Minted per
# process, never persisted (see DECISION.md).
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
    # exception to "every mutating host route is operator-only"
    # (tasks/20260729-125046).
    r"|^/api/host/digests/run/?$"
)


def operator_only(path: str) -> bool:
    """Whether ``path`` demands a real operator session, not a machine token."""
    return OPERATOR_ONLY_PATTERN.match(path) is not None


# Hostnames/addresses that mean "this machine only". A bind to any of these is
# not reachable from the network, which is what makes open development mode
# defensible.
_LOOPBACK_HOSTS: frozenset[str] = frozenset(
    {"127.0.0.1", "localhost", "::1", "[::1]", "0:0:0:0:0:0:0:1"}
)

# scrypt cost. n=2**15 with r=8 needs ~32MB per hash, which is a fine price on a
# login that happens once a session and a real cost to an offline attacker. The
# parameters are encoded INTO the hash so they can be raised later without
# invalidating hashes that already exist.
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16


class AuthConfigError(RuntimeError):
    """The authentication configuration cannot serve safely.

    Raised at app construction, never at request time: a misconfigured deployment
    must fail to start rather than start and quietly serve open.
    """


# --- passwords --------------------------------------------------------------


def hash_password(password: str) -> str:
    """Return an encoded ``scrypt`` hash of ``password``.

    Format: ``scrypt$<n>$<r>$<p>$<salt-b64>$<hash-b64>``. Parameters travel with
    the hash so verification never has to guess them.
    """
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
        maxmem=_SCRYPT_N * _SCRYPT_R * 2 * 64 + 1024 * 1024,
    )
    return "$".join(
        (
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(derived).decode("ascii"),
        )
    )


def verify_password(password: str, encoded: str) -> bool:
    """Whether ``password`` matches the encoded hash. Never raises.

    A malformed, truncated, or foreign-format hash returns False: the failure
    mode of a corrupt credential must be "nobody gets in", not a 500 and not an
    accidental match.
    """
    try:
        scheme, raw_n, raw_r, raw_p, raw_salt, raw_hash = encoded.split("$")
        if scheme != "scrypt":
            return False
        n, r, p = int(raw_n), int(raw_r), int(raw_p)
        salt = base64.b64decode(raw_salt, validate=True)
        expected = base64.b64decode(raw_hash, validate=True)
        if not salt or not expected:
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            maxmem=n * r * 2 * 64 + 1024 * 1024,
        )
    except (ValueError, TypeError, MemoryError, binascii.Error) as exc:
        logger.warning(
            "auth: stored password hash is unusable (%s)", type(exc).__name__
        )
        return False
    return hmac.compare_digest(derived, expected)


# --- policy -----------------------------------------------------------------


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
    reopen exactly the hole this module exists to close.
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
        # is about there being someone to attribute an approval to. Review round
        # 1, finding R1.1.
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


# --- sessions ---------------------------------------------------------------


@dataclass(frozen=True)
class Session:
    """One authenticated session: an opaque id plus its bound CSRF token."""

    id: str
    csrf: str
    created_at: float
    last_seen: float


class SessionStore:
    """Revocable server-side sessions, persisted as JSON under the state dir.

    Persisted (rather than in-memory) so restarting the server does not log the
    operator out - the deployed service restarts on every ``nixos-rebuild
    switch``, and this epic is about to add a button that does exactly that.

    The file is written 0600. It is not a secret store in the sops sense: it
    holds live session ids, readable by the uid the service already runs as.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()
        self._sessions: dict[str, dict[str, float | str]] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            # A corrupt session file logs everyone out; it never crashes startup
            # and never authenticates anyone.
            logger.warning("auth: session store unreadable (%s); starting empty", exc)
            return
        sessions = raw.get("sessions") if isinstance(raw, dict) else None
        if isinstance(sessions, dict):
            self._sessions = {
                sid: rec for sid, rec in sessions.items() if isinstance(rec, dict)
            }

    def _flush(self) -> None:
        """Write the store atomically at 0600.

        ``os.open`` with the mode creates the temp file private from the start;
        writing then chmod-ing would leave a window where it is world-readable.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = json.dumps({"sessions": self._sessions}, indent=2)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, self._path)

    def prune(self, *, now: float, idle: float, absolute: float) -> int:
        """Drop every expired record; return how many went.

        Expiry is otherwise only noticed when an id is PRESENTED, so a session
        whose browser cleared its cookies (or whose device is gone) would sit in
        the file forever, being rewritten on every request. Called on load and
        on create. Review round 1, finding 6.
        """
        with self._lock:
            dead = [
                sid
                for sid, record in self._sessions.items()
                if now - float(record.get("last_seen", 0.0)) > idle
                or now - float(record.get("created_at", 0.0)) > absolute
            ]
            for sid in dead:
                del self._sessions[sid]
            if dead:
                self._flush()
        return len(dead)

    def create(self, *, now: float) -> Session:
        """Mint a new session with a fresh id and CSRF token."""
        session = Session(
            id=secrets.token_urlsafe(32),
            csrf=secrets.token_urlsafe(32),
            created_at=now,
            last_seen=now,
        )
        with self._lock:
            self._sessions[session.id] = {
                "csrf": session.csrf,
                "created_at": now,
                "last_seen": now,
            }
            self._flush()
        return session

    def get(
        self, session_id: str | None, *, now: float, idle: float, absolute: float
    ) -> Session | None:
        """Return the live session for ``session_id``, renewing its idle clock.

        Returns None for an unknown id, an idle-expired session, or one past the
        absolute cap - and drops the expired record on the way out, so expiry is
        real rather than merely reported.
        """
        if not session_id:
            return None
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            created = float(record.get("created_at", 0.0))
            last_seen = float(record.get("last_seen", 0.0))
            if now - last_seen > idle or now - created > absolute:
                del self._sessions[session_id]
                self._flush()
                return None
            record["last_seen"] = now
            self._flush()
            return Session(
                id=session_id,
                csrf=str(record.get("csrf", "")),
                created_at=created,
                last_seen=now,
            )

    def revoke(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._lock:
            if self._sessions.pop(session_id, None) is not None:
                self._flush()

    def revoke_all(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._flush()


# --- login throttling -------------------------------------------------------


class LoginThrottle:
    """Failed-login lockout: per source address, plus a global ceiling.

    Not a defense against a determined attacker on a hostile network - the
    DECISION.md deployment boundary is explicit that public exposure is
    unsupported. It is here so a curious device on the LAN cannot grind the
    password, and so a scripted attempt is slowed to a crawl.

    A fixed window and a hard lockout, NOT a growing delay - stated plainly
    because an earlier version of this docstring claimed more than it did.

    The GLOBAL ceiling exists because per-source alone is trivially evaded: a
    machine with an IPv6 /64 has more addresses than the attacker needs, and each
    one would get its own fresh allowance while the dict of source entries grew
    without bound. Entries are pruned as they age, so an unauthenticated caller
    cannot grow this memory indefinitely either. Review round 1, finding 4.
    """

    # The global ceiling as a multiple of the per-source one. Loose enough that
    # the single operator fat-fingering their password from a phone and a laptop
    # in the same window is unaffected; tight enough to bound a distributed
    # guessing run.
    _GLOBAL_FACTOR = 5

    def __init__(self, *, max_failures: int, window_seconds: float) -> None:
        self._max = max_failures
        self._window = window_seconds
        self._failures: dict[str, list[float]] = {}
        self._lock = Lock()

    def _recent(self, source: str, now: float) -> list[float]:
        return [at for at in self._failures.get(source, []) if now - at < self._window]

    def _prune(self, now: float) -> None:
        """Drop aged-out timestamps and the source entries that empty out.

        Called on every write, so the dict tracks only sources that failed inside
        the current window rather than every address ever seen.
        """
        for source in list(self._failures):
            recent = self._recent(source, now)
            if recent:
                self._failures[source] = recent
            else:
                del self._failures[source]

    def _total(self) -> int:
        return sum(len(times) for times in self._failures.values())

    def allowed(self, source: str, *, now: float) -> bool:
        with self._lock:
            self._prune(now)
            if self._total() >= self._max * self._GLOBAL_FACTOR:
                return False
            return len(self._recent(source, now)) < self._max

    def record_failure(self, source: str, *, now: float) -> None:
        with self._lock:
            self._prune(now)
            self._failures[source] = [*self._recent(source, now), now]

    def record_success(self, source: str) -> None:
        with self._lock:
            self._failures.pop(source, None)

    def retry_after(self, source: str, *, now: float) -> int:
        """Seconds until this source may try again (for the Retry-After header)."""
        with self._lock:
            recent = self._recent(source, now)
            oldest = min(
                (times[0] for times in self._failures.values() if times),
                default=None,
            )
            over_global = self._total() >= self._max * self._GLOBAL_FACTOR
        if len(recent) < self._max and not over_global:
            return 0
        # Whichever bound is holding this caller: their own oldest failure, or the
        # oldest failure anywhere when the global ceiling is what tripped.
        first = min(recent) if recent else oldest
        if first is None:
            return 0
        return max(1, int(self._window - (now - first)))


# --- the machine token ------------------------------------------------------


def mint_api_token() -> str:
    """A fresh per-process machine token for the app's own tool subprocesses.

    Never persisted and never configurable: it dies with the process, so there is
    no credential at rest and no rotation to get wrong. It authenticates
    subprocesses that already run with the operator's privileges, so it grants
    nothing they did not have.
    """
    return secrets.token_urlsafe(32)


def token_matches(presented: str | None, expected: str | None) -> bool:
    """Constant-time equality for a credential, total over ANY input string.

    Both arguments come straight off the wire (an ``Authorization`` or CSRF
    header), and Starlette decodes headers as latin-1, so a raw byte above 0x7F
    reaches this function as a non-ASCII ``str``. ``hmac.compare_digest`` raises
    ``TypeError`` on a non-ASCII ``str``, which would turn a garbage header from
    an UNAUTHENTICATED caller into a 500 and a traceback in the journal. Encoding
    first makes every input compare false instead of raising, and keeps the
    comparison constant-time. ``surrogatepass`` rather than ``surrogateescape``
    so even a lone surrogate encodes rather than raising - the point is that this
    function is TOTAL, not that it round-trips. Caught in review round 1.
    """
    if not presented or not expected:
        return False
    return hmac.compare_digest(
        presented.encode("utf-8", "surrogatepass"),
        expected.encode("utf-8", "surrogatepass"),
    )


def bearer_token(authorization: str | None) -> str | None:
    """Extract the token from an ``Authorization: Bearer <token>`` header."""
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


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


def now() -> float:
    """Current wall clock, indirected so tests can freeze it."""
    return time.time()
