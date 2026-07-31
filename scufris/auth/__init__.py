"""Single-operator authentication for the dashboard.

Scufris is deployed to the LAN with no HTTP authentication at all, which the
host-operator work cannot build on. This package is the whole mechanism:

- a password verified against a stdlib ``scrypt`` hash delivered through the
  same sops dotenv the Telegram token already rides,
- an opaque session id in an ``HttpOnly`` cookie, backed by a revocable
  server-side record under ``state_dir``,
- a per-session CSRF token for the double-submit check,
- failed-login throttling.

The enforcement point itself lives in ``app.py`` as ONE middleware, deny by
default; this package owns the primitives and the policy questions (is
authentication required at all, which paths are public). The session is
server-side rather than a signed cookie, and loopback is not implicitly
trusted.

| Module | Owns |
|--------|------|
| `policy` | every question the middleware asks, including the public-path lists |
| `credentials` | the operator's password hash, and the machine token |
| `store` | revocable session records, login throttling, and the clock |

Nothing here logs a password, a session id, or the machine token.

This module is the package's public surface; the submodules import each other
directly rather than through it.
"""

from __future__ import annotations

from .credentials import (
    bearer_token,
    hash_password,
    mint_api_token,
    token_matches,
    verify_password,
)
from .policy import (
    API_TOKEN_ENV,
    CSRF_COOKIE,
    CSRF_HEADER,
    OPERATOR_ONLY_PATTERN,
    PUBLIC_PATHS,
    PUBLIC_STATIC_PATHS,
    SESSION_COOKIE,
    UNSAFE_METHODS,
    AuthConfigError,
    auth_required,
    is_loopback_host,
    operator_only,
    safe_next_path,
    same_origin,
    session_cookie_kwargs,
    validate_auth_config,
)
from .store import LoginThrottle, Session, SessionStore, now

__all__ = [
    "API_TOKEN_ENV",
    "CSRF_COOKIE",
    "CSRF_HEADER",
    "OPERATOR_ONLY_PATTERN",
    "PUBLIC_PATHS",
    "PUBLIC_STATIC_PATHS",
    "SESSION_COOKIE",
    "UNSAFE_METHODS",
    "AuthConfigError",
    "LoginThrottle",
    "Session",
    "SessionStore",
    "auth_required",
    "bearer_token",
    "hash_password",
    "is_loopback_host",
    "mint_api_token",
    "now",
    "operator_only",
    "safe_next_path",
    "same_origin",
    "session_cookie_kwargs",
    "token_matches",
    "validate_auth_config",
    "verify_password",
]
