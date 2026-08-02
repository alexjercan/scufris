"""The host-check SCHEDULER: when a pass fires, and what a restart does to it.

Driven directly with an injected clock and a fake run - what the scheduler is is
timing and judgement, so nothing here waits or touches the host. What a fired pass
DOES (render, deliver, escalate) is `test_host_digest.py`.

The state lives in the state database now, so the restart proofs REOPEN the file
rather than sharing a handle: the bug this guards against - a bound or an armed
window that only ever existed in one process's memory - is invisible to a test
that never closes anything.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Callable

import pytest

from scufris.db import Database, open_database, upgrade_to_head
from scufris.scheduler import (
    DAILY,
    WATCH,
    HostScheduler,
    SchedulerStore,
    next_daily_due,
)

# A Monday at 09:30 local, so "08:00" is always in the past for the same day and the
# arithmetic below is readable.
NOW = time.mktime(time.struct_time((2026, 7, 27, 9, 30, 0, 0, 208, -1)))


def _clock(start: float = NOW) -> tuple[Callable[[], float], Callable[[float], None]]:
    """A clock the test moves by hand, so nothing here sleeps."""
    state = {"t": start}

    def now() -> float:
        return state["t"]

    def advance(seconds: float) -> None:
        state["t"] += seconds

    return now, advance


async def test_a_schedule_fires_when_it_is_due_and_not_before(
    database: Database,
) -> None:
    now, advance = _clock()
    ran: list[str] = []

    async def run(name: str) -> str:
        ran.append(name)
        return "ran"

    scheduler = HostScheduler(
        SchedulerStore(database),
        run=run,
        watch_interval=lambda: 900.0,
        daily_at=lambda: "08:00",
        watch_enabled=lambda: True,
        daily_enabled=lambda: True,
        muted_until=lambda: 0.0,
        clock=now,
    )

    # First sight ARMS both schedules and runs nothing: a fresh boot (or a restart
    # loop) must not fire a pass at startup.
    assert await scheduler.tick() == []
    assert ran == []
    states = {state.name: state for state in await scheduler.states()}
    assert states[WATCH].next_due == pytest.approx(NOW + 900)
    assert states[DAILY].next_due == pytest.approx(next_daily_due("08:00", now=NOW))
    assert "nothing has run yet" in states[WATCH].last_result

    # Not yet due: still nothing.
    advance(899)
    assert await scheduler.tick() == []

    # Due: it runs once, and re-arms.
    advance(2)
    assert await scheduler.tick() == [WATCH]
    assert ran == [WATCH]
    watch = {s.name: s for s in await scheduler.states()}[WATCH]
    assert watch.runs == 1
    assert watch.last_run == pytest.approx(NOW + 901)
    assert watch.next_due == pytest.approx(NOW + 901 + 900)


async def test_schedules_survive_restart_without_stampede(tmp_path: Path) -> None:
    """A window missed while the app was down is recorded, not replayed.

    The failure this pins: an app down for six hours coming back and delivering
    twenty-four `watch` digests at once, which is how a useful feature becomes a
    muted one.
    """
    now, advance = _clock()
    ran: list[str] = []

    async def run(name: str) -> str:
        ran.append(name)
        return "ran"

    def build(db: Database, clock: Callable[[], float]) -> HostScheduler:
        return HostScheduler(
            SchedulerStore(db),
            run=run,
            watch_interval=lambda: 900.0,
            daily_at=lambda: "08:00",
            watch_enabled=lambda: True,
            daily_enabled=lambda: True,
            muted_until=lambda: 0.0,
            clock=clock,
        )

    # Two handles opened in sequence rather than one shared one: a restart REOPENS
    # the file, and a shared handle would prove only that two objects can see one
    # connection pool.
    # Offloaded: this test is `async def`, and the migration opens transactions.
    db = open_database(tmp_path)
    await asyncio.to_thread(upgrade_to_head, db)
    try:
        first = build(db, now)
        await first.tick()  # arms
        advance(901)
        await first.tick()  # one real run
        assert ran == [WATCH]
        armed = {s.name: s.next_due for s in await first.states()}
    finally:
        db.close()

    # The app goes away for six hours and comes back. A SECOND scheduler over the
    # same state dir is what a restart is.
    advance(6 * 3600)
    db = open_database(tmp_path)
    try:
        second = build(db, now)
        restored = {s.name: s for s in await second.states()}
        assert restored[WATCH].next_due == pytest.approx(armed[WATCH])
        assert restored[WATCH].runs == 1  # the history survived

        ran.clear()
        # The window is long past: skipped, not fired.
        assert await second.tick() == []
        assert ran == []
        watch = {s.name: s for s in await second.states()}[WATCH]
        assert watch.missed == 1
        assert "window missed" in watch.last_result
        assert watch.next_due == pytest.approx(now() + 900)

        # And the next real window runs exactly once - no backlog.
        advance(901)
        assert await second.tick() == [WATCH]
        assert ran == [WATCH]
    finally:
        db.close()


async def test_a_run_never_overlaps_itself(database: Database) -> None:
    now, advance = _clock()
    release = asyncio.Event()
    started = asyncio.Event()
    calls: list[str] = []

    async def run(name: str) -> str:
        calls.append(name)
        started.set()
        await release.wait()
        return "ran"

    scheduler = HostScheduler(
        SchedulerStore(database),
        run=run,
        watch_interval=lambda: 900.0,
        daily_at=lambda: "08:00",
        watch_enabled=lambda: True,
        daily_enabled=lambda: False,
        muted_until=lambda: 0.0,
        clock=now,
    )
    await scheduler.tick()  # arms
    advance(901)
    first = asyncio.create_task(scheduler.tick())
    await asyncio.wait_for(started.wait(), timeout=5)

    # A second tick while the first run is in flight records a skip and does NOT
    # start a second pass over the same subprocess reads.
    advance(901)
    assert await scheduler.tick() == []
    assert calls == [WATCH]
    skipped = {s.name: s for s in await scheduler.states()}[WATCH]
    assert "previous run was still going" in skipped.last_result
    assert skipped.missed == 1

    release.set()
    await asyncio.wait_for(first, timeout=5)
    assert calls == [WATCH]


async def test_a_disabled_schedule_does_nothing_and_a_failing_run_is_recorded(
    database: Database,
) -> None:
    now, advance = _clock()
    enabled = {"watch": False}

    async def run(name: str) -> str:
        raise RuntimeError("the checks exploded")

    scheduler = HostScheduler(
        SchedulerStore(database),
        run=run,
        watch_interval=lambda: 900.0,
        daily_at=lambda: "08:00",
        watch_enabled=lambda: enabled["watch"],
        daily_enabled=lambda: False,
        muted_until=lambda: 0.0,
        clock=now,
    )
    advance(10_000)
    assert await scheduler.tick() == []  # disabled: not even armed into running

    enabled["watch"] = True
    await scheduler.tick()  # arms
    advance(901)
    assert await scheduler.tick() == [WATCH]
    # A run that raises is RECORDED, not propagated: a scheduler that dies on one bad
    # pass is silent in exactly the way this feature exists to prevent.
    state = {s.name: s for s in await scheduler.states()}[WATCH]
    assert "the run failed" in state.last_result
    assert "the checks exploded" in state.last_result
    assert state.next_due > now()


def test_the_daily_time_is_local_and_a_typo_does_not_disable_it() -> None:
    due = next_daily_due("08:00", now=NOW)
    assert due > NOW  # 09:30 is past 08:00, so it is tomorrow
    assert time.localtime(due).tm_hour == 8
    assert time.localtime(due).tm_min == 0
    later = next_daily_due("23:45", now=NOW)
    assert time.localtime(later).tm_hour == 23
    # A malformed value costs the hour, not the heartbeat.
    assert time.localtime(next_daily_due("nonsense", now=NOW)).tm_hour == 8
