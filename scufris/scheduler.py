"""The one proactive thing Scufris does: fire the host checks on a schedule.

Everything else here starts from a person - the UI, a chat, Telegram. `wake.py`
wakes the ORCHESTRATOR when a sub-agent needs it; this wakes the OPERATOR.

Two schedules, with fixed identities rather than an operator-defined schedule
language:

- ``watch`` - every ``host_watch_interval_seconds``, and it only DELIVERS when a
  check wants attention or something recovered;
- ``daily`` - once a day at ``host_digest_at``, always delivering, even if it is one
  line. That line is the heartbeat: it is what makes silence from ``watch``
  unambiguous.

Three properties the loop has to have, and each one is a way this feature could
otherwise become a nuisance rather than a help:

- **State persists.** ``next_due``, ``last_run``, ``last_result`` survive a restart,
  so a bounce does not reset the day.
- **A missed window is recorded, not fired.** An app that was down for six hours must
  not deliver twenty-four `watch` digests when it comes back. The window is skipped
  with the miss counted, and the next one is scheduled from NOW.
- **No overlap.** A schedule whose previous run is still going records a skip rather
  than starting a second pass over the same subprocess-backed reads.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

from pydantic import BaseModel
from sqlalchemy import Connection, insert, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .db import Database
from .db.models import ScheduleRow

logger = logging.getLogger(__name__)

WATCH = "watch"
DAILY = "daily"

# How often the loop looks at the clock. Fine-grained enough that a time-of-day
# schedule fires within the minute it was asked for, coarse enough to be free.
TICK_SECONDS = 20.0


class ScheduleState(BaseModel):
    """One schedule's persisted state."""

    name: str
    # Unix time this schedule is next allowed to run. 0 means "not scheduled yet":
    # the loop sets it one full window ahead the first time it sees this schedule.
    #
    # It does NOT run on sight, and that was a build-time correction: firing on a
    # fresh schedule made every app start - including every test that boots one -
    # perform real subprocess reads of the host, and made a restart loop a way to
    # hammer the machine. Nothing proactive fires within the first window of a boot;
    # the operator's run-now button is how the feature proves itself.
    next_due: float = 0.0
    last_run: float | None = None
    # A short sentence: what happened last time. This is what the dashboard shows
    # when the answer to "did it fire" was silence.
    last_result: str = ""
    # How many windows were skipped because the app was down or a run was in flight.
    missed: int = 0
    runs: int = 0


class SchedulerStore:
    """The schedules' state, in the state database.

    Each method is ONE unit of work on the app's one transactional boundary
    (``packages/core/src/scufris_core/engine.py``), and every one is SYNCHRONOUS:
    ``HostScheduler``'s methods are ``async def``, so they offload these calls
    with ``asyncio.to_thread`` rather than holding SQLite's write lock under the
    loop.

    There is no in-memory mirror. A ``ScheduleState`` this returns is a detached
    copy: mutating it changes nothing until it is handed back to :meth:`save`,
    which is what the scheduler does at the end of every branch that touches one.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, name: str) -> ScheduleState:
        """This schedule's state, creating it on first sight.

        The read and the create-on-read are ONE transaction (they were a read
        outside the persist before): every begin on this engine is immediate, so
        two processes seeing a schedule for the first time cannot both insert it.
        """
        with self._db.transaction() as conn:
            return _get_or_create(conn, name)

    def save(self, state: ScheduleState) -> None:
        with self._db.transaction() as conn:
            _write(conn, state)

    def all(self) -> list[ScheduleState]:
        """Both schedules, in their fixed order, each created on first sight."""
        with self._db.transaction() as conn:
            return [_get_or_create(conn, name) for name in (WATCH, DAILY)]


def _get_or_create(conn: Connection, name: str) -> ScheduleState:
    """One schedule's row on an OPEN connection, inserted if it is not there yet."""
    row = conn.execute(
        select(ScheduleRow.__table__).where(ScheduleRow.name == name)
    ).first()
    if row is not None:
        return ScheduleState.model_validate(dict(row._mapping))
    state = ScheduleState(name=name)
    conn.execute(insert(ScheduleRow).values(**state.model_dump()))
    return state


def _write(conn: Connection, state: ScheduleState) -> None:
    """Upsert one schedule on an OPEN connection.

    An upsert rather than an update: ``save`` is also how a schedule the process
    has never read gets its first row, and the name is the primary key.
    """
    values = state.model_dump()
    conn.execute(
        sqlite_insert(ScheduleRow)
        .values(**values)
        .on_conflict_do_update(index_elements=[ScheduleRow.name], set_=values)
    )


# What a run of the checks does, injected so the scheduler owns timing and nothing
# else: it takes the schedule name and returns the sentence to record.
RunChecks = Callable[[str], Awaitable[str]]


def next_daily_due(at: str, *, now: float, clock_tz: bool = True) -> float:
    """The next unix time matching ``at`` ("HH:MM") in the LOCAL zone.

    Local, not UTC, because "08:00" means the operator's morning. A malformed value
    falls back to 08:00 rather than disabling the schedule silently - a typo in a
    settings field should cost a wrong hour, not the whole heartbeat.
    """
    try:
        hour_text, _, minute_text = at.partition(":")
        hour, minute = int(hour_text), int(minute_text)
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError(at)
    except ValueError:
        logger.warning("host_digest_at %r is not HH:MM; using 08:00", at)
        hour, minute = 8, 0
    moment = datetime.fromtimestamp(now).astimezone()
    target = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= moment:
        target += timedelta(days=1)
    return target.timestamp()


class HostScheduler:
    """Fires the host checks on the two schedules, and records what happened.

    Owns no checks and no delivery: ``run`` is injected, so this class is about the
    clock and the bookkeeping. That is also what makes it testable without waiting -
    every test drives ``tick`` with an injected clock rather than sleeping.
    """

    def __init__(
        self,
        store: SchedulerStore,
        *,
        run: RunChecks,
        watch_interval: Callable[[], float],
        daily_at: Callable[[], str],
        watch_enabled: Callable[[], bool],
        daily_enabled: Callable[[], bool],
        muted_until: Callable[[], float],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._run = run
        self._watch_interval = watch_interval
        self._daily_at = daily_at
        self._watch_enabled = watch_enabled
        self._daily_enabled = daily_enabled
        self._muted_until = muted_until
        self._now = clock
        # Which schedules have a run in flight. The overlap guard; also why `tick`
        # can be called as often as anyone likes.
        self._running: set[str] = set()
        # Manual runs in flight, held so a fire-and-forget task is not garbage
        # collected mid-run. It lives here rather than in the route that starts
        # one: the task outlives the request by design, so whatever keeps it
        # alive has to outlive the request too.
        self._manual: set[asyncio.Task[None]] = set()

    # --- reads ------------------------------------------------------------

    async def states(self) -> list[ScheduleState]:
        """Both schedules' state. Offloaded: the store opens a transaction, and
        every caller of this is on the event loop."""
        return await asyncio.to_thread(self._store.all)

    @property
    def store(self) -> SchedulerStore:
        """The store behind this scheduler, for the boundary proof and for tests."""
        return self._store

    def muted(self) -> bool:
        return self._now() < self._muted_until()

    # --- the loop ---------------------------------------------------------

    async def run_forever(self, *, tick_seconds: float = TICK_SECONDS) -> None:
        """Tick until cancelled. The app's lifespan owns this task."""
        logger.info("host check scheduler started")
        try:
            while True:
                try:
                    await self.tick()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A tick that blows up must not end the scheduler: the next one
                    # may well work, and a dead scheduler is silent in exactly the
                    # way this feature exists to avoid.
                    logger.exception("host check tick failed; continuing")
                await asyncio.sleep(tick_seconds)
        finally:
            logger.info("host check scheduler stopped")

    async def tick(self) -> list[str]:
        """Run whatever is due. Returns the names that ran (for tests and run-now)."""
        ran: list[str] = []
        for name in (WATCH, DAILY):
            if await self._maybe_run(name):
                ran.append(name)
        return ran

    async def run_now(self, name: str) -> str:
        """Run one schedule immediately, regardless of when it was due.

        The operator's "run it now" button. It still respects the overlap guard - two
        passes over the same subprocess reads is not what anyone means by now - and it
        still moves the schedule's next due time, so a manual run counts as the run.
        """
        if name not in (WATCH, DAILY):
            raise ValueError(f"no such schedule: {name}")
        if name in self._running:
            return "a run of this schedule is already in flight"
        state = await asyncio.to_thread(self._store.get, name)
        return await self._execute(name, state)

    def start_now(self, name: str) -> None:
        """Start one schedule's run in the background, and return immediately.

        The "run it now" button. It does NOT wait: a full pass walks the nix
        store and shells out to systemctl, which is tens of seconds of work that
        no HTTP request should hold a connection open for - the caller polls the
        digests for the result. The overlap guard in ``run_now`` still applies,
        so this cannot be used to run concurrent passes over the same reads.

        Raises ``ValueError`` for an unknown schedule, BEFORE anything is
        started, so a caller can refuse rather than report a run that is not
        happening.
        """
        if name not in (WATCH, DAILY):
            raise ValueError(f"no such schedule: {name}")

        async def _run() -> None:
            try:
                await self.run_now(name)
            except Exception:  # noqa: BLE001 - recorded on the schedule already
                logger.exception("the manual %s run failed", name)

        task = asyncio.create_task(_run())
        self._manual.add(task)
        task.add_done_callback(self._manual.discard)

    # --- internals --------------------------------------------------------

    def _enabled(self, name: str) -> bool:
        return self._watch_enabled() if name == WATCH else self._daily_enabled()

    def _interval(self, name: str) -> float:
        if name == WATCH:
            return max(60.0, self._watch_interval())
        return 24 * 3600.0

    def _schedule_next(self, name: str, state: ScheduleState, *, now: float) -> None:
        """Point a schedule at its next window, FROM NOW.

        From now rather than from the window it missed: that is what stops a
        stampede. An app down for six hours comes back, records the misses, and
        resumes on the next real window instead of replaying the day.
        """
        if name == DAILY:
            state.next_due = next_daily_due(self._daily_at(), now=now)
        else:
            state.next_due = now + self._interval(name)

    async def _maybe_run(self, name: str) -> bool:
        # Every store call in this method is offloaded: `tick` runs on the loop,
        # and a transaction opened there would hold SQLite's write lock under it.
        state = await asyncio.to_thread(self._store.get, name)
        now = self._now()
        if not self._enabled(name):
            return False
        if state.next_due == 0.0:
            # First sight of this schedule: arm it one window ahead and run nothing.
            self._schedule_next(name, state, now=now)
            state.last_result = "scheduled; nothing has run yet"
            await asyncio.to_thread(self._store.save, state)
            logger.info(
                "host schedule %s armed for %s", name, _stamp_local(state.next_due)
            )
            return False
        if now < state.next_due:
            return False
        # In flight FIRST, before the lateness classification. A run that outlives
        # half an interval would otherwise be reported as a window that passed while
        # nobody was home - which is the opposite of what happened, and the wrong
        # sentence to leave in the record. Caught by
        # `test_a_run_never_overlaps_itself`.
        if name in self._running:
            state.missed += 1
            state.last_result = "skipped: the previous run was still going"
            self._schedule_next(name, state, now=now)
            await asyncio.to_thread(self._store.save, state)
            logger.info("host schedule %s skipped: previous run still in flight", name)
            return False
        # The window is due. How LATE it is decides whether it is a real window or a
        # window that passed while nobody was home.
        lateness = now - state.next_due
        if lateness > self._grace(name):
            state.missed += 1
            state.last_result = (
                f"window missed by {_duration(lateness)} (the app was not running); "
                "skipped rather than replayed"
            )
            self._schedule_next(name, state, now=now)
            await asyncio.to_thread(self._store.save, state)
            logger.info("host schedule %s missed a window by %.0fs", name, lateness)
            return False
        return await self._start(name, state, now)

    def _grace(self, name: str) -> float:
        """How late a window may be and still count as that window.

        Half an interval for `watch` (a tick that slipped), and two hours for the
        daily one (a laptop lid, a restart over breakfast) - after that the morning
        has passed and a digest about it would be about a machine that has moved on.
        """
        return self._interval(name) / 2 if name == WATCH else 2 * 3600.0

    async def _start(self, name: str, state: ScheduleState, now: float) -> bool:
        # The overlap guard lives in `_maybe_run` (before the lateness check) and in
        # `run_now`; by here the schedule is known not to be running.
        await self._execute(name, state)
        return True

    async def _execute(self, name: str, state: ScheduleState) -> str:
        """Run the checks for one schedule and record the outcome, whatever it is."""
        self._running.add(name)
        started = self._now()
        try:
            result = await self._run(name)
        except asyncio.CancelledError:
            state.last_result = "cancelled (the app was shutting down)"
            await self._finish(name, state, started)
            raise
        except Exception as exc:  # noqa: BLE001 - a failed run is recorded, not raised
            logger.exception("host schedule %s failed", name)
            result = f"the run failed: {type(exc).__name__}: {exc}"
        state.last_result = result
        state.runs += 1
        await self._finish(name, state, started)
        return result

    async def _finish(self, name: str, state: ScheduleState, started: float) -> None:
        state.last_run = started
        self._schedule_next(name, state, now=self._now())
        await asyncio.to_thread(self._store.save, state)
        self._running.discard(name)


def _stamp_local(at: float) -> str:
    return datetime.fromtimestamp(at).astimezone().strftime("%Y-%m-%d %H:%M")


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"
