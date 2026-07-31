"""Server-side session records and failed-login throttling, and the clock both use.

Sessions are server-side rather than a signed cookie so they are REVOCABLE: an
id in a cookie is worthless once the record behind it is gone.

Nothing here logs a session id.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


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
    switch``, which is a button this dashboard offers.

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
        on create.
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


class LoginThrottle:
    """Failed-login lockout: per source address, plus a global ceiling.

    Not a defense against a determined attacker on a hostile network - public
    exposure is an unsupported deployment. It is here so a curious device on the
    LAN cannot grind the password, and so a scripted attempt is slowed to a
    crawl.

    A fixed window and a hard lockout, NOT a growing delay - stated plainly
    because an earlier version of this docstring claimed more than it did.

    The GLOBAL ceiling exists because per-source alone is trivially evaded: a
    machine with an IPv6 /64 has more addresses than the attacker needs, and each
    one would get its own fresh allowance while the dict of source entries grew
    without bound. Entries are pruned as they age, so an unauthenticated caller
    cannot grow this memory indefinitely either.
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


def now() -> float:
    """Current wall clock, indirected so tests can freeze it."""
    return time.time()
