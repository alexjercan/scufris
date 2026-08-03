"""Server-side session records and failed-login throttling, and the clock both use.

Sessions are server-side rather than a signed cookie so they are REVOCABLE: an
id in a cookie is worthless once the record behind it is gone.

They live in the state database, on the one transactional boundary
(``packages/core/src/scufris_core/engine.py``). Every method here is ONE unit of
work, which is what makes ``get``'s read-renew-expire atomic rather than a read
followed by a write that something else can land between. There is no in-memory
mirror and no lock of this module's own: SQLite's write lock is the lock.

Every method is SYNCHRONOUS and opens a transaction, so an ``async def`` caller
must offload it with ``asyncio.to_thread`` - the engine refuses a thread with a
running event loop rather than holding the write lock under it.

Nothing here logs a session id.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass
from threading import Lock

from sqlalchemy import delete as sql_delete
from sqlalchemy import insert, or_, select
from sqlalchemy import update as sql_update

from ..db import Database
from ..db.models import AuthSessionRow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Session:
    """One authenticated session: an opaque id plus its bound CSRF token."""

    id: str
    csrf: str
    created_at: float
    last_seen: float


class SessionStore:
    """Revocable server-side sessions, in the state database.

    Persisted (rather than in-memory) so restarting the server does not log the
    operator out - the deployed service restarts on every ``nixos-rebuild
    switch``, which is a button this dashboard offers.

    The database file is 0600, sidecars included
    (``packages/core/src/scufris_core/engine.py``). It is not a secret store in
    the sops sense: it holds live session ids, readable by the uid the service
    already runs as.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def prune(self, *, now: float, idle: float, absolute: float) -> int:
        """Drop every expired record; return how many went.

        Expiry is otherwise only noticed when an id is PRESENTED, so a session
        whose browser cleared its cookies (or whose device is gone) would sit in
        the table forever, being renewed by nobody and read by every sweep.
        Called at startup and on login.
        """
        with self._db.transaction() as conn:
            result = conn.execute(
                sql_delete(AuthSessionRow).where(
                    or_(
                        AuthSessionRow.last_seen < now - idle,
                        AuthSessionRow.created_at < now - absolute,
                    )
                )
            )
            return result.rowcount

    def create(self, *, now: float) -> Session:
        """Mint a new session with a fresh id and CSRF token."""
        session = Session(
            id=secrets.token_urlsafe(32),
            csrf=secrets.token_urlsafe(32),
            created_at=now,
            last_seen=now,
        )
        with self._db.transaction() as conn:
            conn.execute(
                insert(AuthSessionRow).values(
                    id=session.id,
                    csrf=session.csrf,
                    created_at=now,
                    last_seen=now,
                )
            )
        return session

    def get(
        self, session_id: str | None, *, now: float, idle: float, absolute: float
    ) -> Session | None:
        """Return the live session for ``session_id``, renewing its idle clock.

        Returns None for an unknown id, an idle-expired session, or one past the
        absolute cap - and drops the expired record on the way out, so expiry is
        real rather than merely reported.

        The read, the expiry check and whichever write follows are ONE unit of
        work. Read-then-write across two transactions would let a concurrent
        revoke land in between and be undone by the renewal, which is a logout
        that did not log anyone out.

        This renews on EVERY authenticated request, so the read path takes the
        write lock (20260801-100413 DECISION.md 3). Taken deliberately: the JSON
        store rewrote the whole session file on the same path, so a single keyed
        update is strictly cheaper.
        """
        if not session_id:
            return None
        with self._db.transaction() as conn:
            row = conn.execute(
                select(AuthSessionRow.__table__).where(AuthSessionRow.id == session_id)
            ).first()
            if row is None:
                return None
            if now - row.last_seen > idle or now - row.created_at > absolute:
                conn.execute(
                    sql_delete(AuthSessionRow).where(AuthSessionRow.id == session_id)
                )
                return None
            conn.execute(
                sql_update(AuthSessionRow)
                .where(AuthSessionRow.id == session_id)
                .values(last_seen=now)
            )
            return Session(
                id=session_id,
                csrf=row.csrf,
                created_at=row.created_at,
                last_seen=now,
            )

    def revoke(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._db.transaction() as conn:
            conn.execute(
                sql_delete(AuthSessionRow).where(AuthSessionRow.id == session_id)
            )

    def revoke_all(self) -> None:
        with self._db.transaction() as conn:
            conn.execute(sql_delete(AuthSessionRow))


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
