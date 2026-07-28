"""The agent-run supervisor: background execution decoupled from HTTP requests.

This replaces the request-scoped model (a turn ran inside the held
``/api/chat/stream`` request under a single global ``chat_lock``, guarded by a
120s timeout) with the ADR-001 model (tasks/20260720-221748/SPIKE.md): a run is
a background task the supervisor owns; the HTTP layer only starts it and relays
its ``EventBus``. Consequences:

- Multiple agents run concurrently under a semaphore (the concurrency cap acts
  as the queue); turns of the SAME agent still serialize via ``serialize_key``.
- A run outlives the request that started it - a dropped SSE relay does not
  cancel it.
- There is no wall-clock request timeout. A background run passes
  ``budget_seconds=None`` (no cap); a per-event ``heartbeat_seconds`` guard
  cancels only a genuinely stalled run (no event for that long).

Serialization uses a SYNCHRONOUS FIFO reservation (``reserve``), not an
``asyncio.Lock`` acquired inside the task: ``start`` reserves the key's slot
before it returns, so a session-mutating endpoint (reset/new/switch) that
reserves the same key afterwards is guaranteed to queue behind the in-flight
turn. Acquiring the lock only once the background task happened to be scheduled
left a window where a reset could slip in front of its own turn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable

from pydantic import BaseModel

from .agent import StreamError, StreamEvent
from .enums import RunPhase
from .eventbus import EventBus

logger = logging.getLogger(__name__)

# A factory producing the run's event stream. It is called once, inside the
# background task, so the agent turn does not start until a concurrency slot is
# free (the factory, not a live iterator, is what gets queued).
MakeStream = Callable[[], AsyncIterator[StreamEvent]]

# A reservation on a serialize key: the predecessor's completion Future to await
# (or None if first in line) and a release callable to run when done.
Reservation = tuple["asyncio.Future[None] | None", Callable[[], None]]


class AgentRunStalled(Exception):
    """A run produced no event within its heartbeat window.

    Internal only - raised solely by the supervisor's heartbeat guard, never by a
    ``make_stream``; so catching it in ``_execute`` cannot misclassify a caller
    exception as a stall.
    """


class RunState(BaseModel):
    """A snapshot of one run's status, safe to serialize to the API."""

    run_id: str
    state: RunPhase
    started_at: float | None = None
    ended_at: float | None = None
    last_event_at: float | None = None
    error: str | None = None
    # True when this run was cancelled by an explicit ``cancel(run_id)`` (the
    # user's stop button or the orchestrator's ``cancel_agent`` tool), as opposed
    # to a stall/budget/crash. The persist callback keys the terminal AgentState
    # off THIS flag, not the ``error`` string, so a user stop reads as a neutral
    # CANCELLED rather than an ERROR (and a real error whose detail happens to be
    # "cancelled" is not misclassified).
    cancelled: bool = False
    # The turn's prompt, so a client reattaching mid-turn can render the user
    # bubble before the backend has flushed it to its durable log. Raw (unsteered);
    # the read boundary strips steering, mirroring read_transcript.
    prompt: str | None = None


class _Run:
    __slots__ = (
        "run_id",
        "make_stream",
        "budget_seconds",
        "heartbeat_seconds",
        "reservation",
        "on_complete",
        "bus",
        "prompt",
        "state",
        "started_at",
        "ended_at",
        "last_event_at",
        "error",
        "cancelled",
        "task",
    )

    def __init__(
        self,
        run_id: str,
        make_stream: MakeStream,
        budget_seconds: float | None,
        heartbeat_seconds: float | None,
        reservation: Reservation,
        on_complete: "Callable[[RunState], None] | None",
        bus: EventBus,
        prompt: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.make_stream = make_stream
        self.budget_seconds = budget_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.reservation = reservation
        self.on_complete = on_complete
        self.bus = bus
        self.prompt = prompt
        self.state: RunPhase = RunPhase.QUEUED
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.last_event_at: float | None = None
        self.error: str | None = None
        self.cancelled: bool = False
        self.task: asyncio.Task[None] | None = None

    def snapshot(self) -> RunState:
        return RunState(
            run_id=self.run_id,
            state=self.state,
            started_at=self.started_at,
            ended_at=self.ended_at,
            last_event_at=self.last_event_at,
            error=self.error,
            cancelled=self.cancelled,
            prompt=self.prompt,
        )


class Supervisor:
    """Owns background agent runs, their event buses, and their lifecycle."""

    def __init__(
        self,
        *,
        max_concurrent: int = 4,
        max_history: int = 200,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)
        self._runs: dict[str, _Run] = {}
        # Per serialize key, the tail of its FIFO reservation chain (the Future
        # the next reserver must await). Cleared when the chain empties.
        self._tails: dict[str, asyncio.Future[None]] = {}
        # Terminal run ids in completion order; the oldest are reaped past the cap
        # so `_runs` cannot grow without bound on a long-lived server.
        self._terminal: deque[str] = deque()
        self._max_history = max_history
        self._now = clock

    def reserve(self, key: str) -> Reservation:
        """Take the next slot in ``key``'s FIFO chain, synchronously.

        Returns the predecessor's completion Future (await it before proceeding,
        or None if first) and a ``release`` to call when done. Being synchronous
        is the point: call order == serialization order, independent of when the
        background task is scheduled.
        """
        loop = asyncio.get_event_loop()
        prev = self._tails.get(key)
        mine: "asyncio.Future[None]" = loop.create_future()
        self._tails[key] = mine

        def release() -> None:
            if not mine.done():
                mine.set_result(None)
            if self._tails.get(key) is mine:
                del self._tails[key]

        return prev, release

    @asynccontextmanager
    async def serialized(self, key: str) -> AsyncIterator[None]:
        """Async context manager form of a reservation, for callers (endpoints).

        Reserves synchronously on entry so ordering relative to a concurrently
        starting run is deterministic, then awaits the predecessor.
        """
        prev, release = self.reserve(key)
        try:
            if prev is not None:
                await prev
            yield
        finally:
            release()

    def start(
        self,
        run_id: str,
        make_stream: MakeStream,
        *,
        serialize_key: str | None = None,
        budget_seconds: float | None = None,
        heartbeat_seconds: float | None = None,
        on_complete: "Callable[[RunState], None] | None" = None,
        prompt: str | None = None,
    ) -> EventBus:
        """Schedule ``make_stream`` as a background run and return its bus.

        The serialize slot (if any) is reserved HERE, synchronously, before this
        returns - so a mutation endpoint reserving the same key afterwards queues
        behind this run. The bus is available immediately so a relay can subscribe
        before the run gets a concurrency slot. ``on_complete`` (if given) is
        invoked with the terminal ``RunState`` after the run ends - the run engine
        uses it to persist the agent's state + session id. ``prompt`` (if given) is
        exposed on the run's status snapshot so a mid-turn reattach can render the
        user bubble.
        """
        if run_id in self._runs:
            raise ValueError(f"run already exists: {run_id}")
        reservation: Reservation = (
            self.reserve(serialize_key)
            if serialize_key is not None
            else (None, lambda: None)
        )
        bus = EventBus()
        run = _Run(
            run_id,
            make_stream,
            budget_seconds,
            heartbeat_seconds,
            reservation,
            on_complete,
            bus,
            prompt=prompt,
        )
        self._runs[run_id] = run
        run.task = asyncio.create_task(self._execute(run), name=f"agent-run:{run_id}")
        return bus

    def bus(self, run_id: str) -> EventBus | None:
        run = self._runs.get(run_id)
        return run.bus if run is not None else None

    def status(self, run_id: str) -> RunState | None:
        run = self._runs.get(run_id)
        return run.snapshot() if run is not None else None

    def list_runs(self) -> list[RunState]:
        return [run.snapshot() for run in self._runs.values()]

    def cancel(self, run_id: str) -> bool:
        """Cancel a live run: mark it cancelled and cancel its task.

        Returns True if a live (not-yet-terminal) run was cancelled, False when
        the run is unknown or already finished. The ``cancelled`` flag is set
        BEFORE cancelling so the terminal snapshot the ``on_complete`` callback
        reads carries it. Cancelling the task raises ``CancelledError`` into the
        drain loop; ``_drain``'s finally ``aclose()``s the backend stream so its
        own cleanup runs (e.g. the Claude backend kills its subprocess), making
        this a real upstream abort, not just a detach. ``_execute`` then settles
        the run terminal and publishes a ``StreamError`` so relays end cleanly.
        """
        run = self._runs.get(run_id)
        if run is None or run.task is None or run.task.done():
            return False
        run.cancelled = True
        run.task.cancel()
        return True

    async def aclose(self) -> None:
        """Cancel every live run (used on app shutdown)."""
        tasks = [run.task for run in self._runs.values() if run.task is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    def _retire(self, run_id: str) -> None:
        """Record a terminal run and reap the oldest beyond the history cap."""
        self._terminal.append(run_id)
        while len(self._terminal) > self._max_history:
            old = self._terminal.popleft()
            self._runs.pop(old, None)

    async def _execute(self, run: _Run) -> None:
        prev, release = run.reservation
        try:
            # Wait our turn in the serialize chain BEFORE taking a concurrency
            # slot, so a burst of one agent's turns queues on its own key instead
            # of starving other agents by holding slots while blocked.
            if prev is not None:
                await prev
            async with self._sem:
                run.state = RunPhase.RUNNING
                run.started_at = self._now()
                run.last_event_at = run.started_at
                await self._drain_with_limits(run)
                run.state = RunPhase.DONE
        except asyncio.CancelledError:
            run.state = RunPhase.ERROR
            run.error = run.error or "cancelled"
            run.bus.publish(StreamError(detail=run.error))
            raise
        except AgentRunStalled as exc:
            run.state = RunPhase.ERROR
            run.error = str(exc)
            logger.warning("agent run %s stalled: %s", run.run_id, exc)
            run.bus.publish(StreamError(detail=run.error))
        except asyncio.TimeoutError:
            run.state = RunPhase.ERROR
            run.error = f"run exceeded budget of {run.budget_seconds}s"
            logger.warning("agent run %s over budget", run.run_id)
            run.bus.publish(StreamError(detail=run.error))
        except Exception as exc:  # noqa: BLE001 - surface any run failure as an event
            run.state = RunPhase.ERROR
            run.error = str(exc)
            logger.exception("agent run %s failed", run.run_id)
            run.bus.publish(StreamError(detail=run.error))
        finally:
            run.ended_at = self._now()
            run.bus.close()
            release()  # let the next same-key run/mutation proceed
            if run.on_complete is not None:
                try:
                    run.on_complete(run.snapshot())
                except Exception:  # noqa: BLE001 - a callback error must not break the supervisor
                    logger.exception("on_complete for run %s failed", run.run_id)
            self._retire(run.run_id)

    async def _drain_with_limits(self, run: _Run) -> None:
        """Run the stream to exhaustion, enforcing the optional total budget.

        The per-event heartbeat lives in ``_drain``; the total budget wraps it so
        even a stream that emits nothing at all is bounded when a budget is set.
        """
        if run.budget_seconds is not None:
            await asyncio.wait_for(self._drain(run), timeout=run.budget_seconds)
        else:
            await self._drain(run)

    async def _drain(self, run: _Run) -> None:
        agen = run.make_stream()
        anext = agen.__anext__
        try:
            while True:
                try:
                    if run.heartbeat_seconds is not None:
                        event = await asyncio.wait_for(
                            anext(), timeout=run.heartbeat_seconds
                        )
                    else:
                        event = await anext()
                except StopAsyncIteration:
                    return
                except asyncio.TimeoutError as exc:
                    raise AgentRunStalled(
                        f"no event within {run.heartbeat_seconds}s"
                    ) from exc
                run.last_event_at = self._now()
                # A backend that ends a turn in failure (idle timeout, over-limit
                # line, thread-setup failure) yields a terminal StreamError and then
                # STOPS - the stream completes normally, so RunPhase settles DONE and
                # the except-clauses in _execute never fire. Record the detail on
                # run.error (last-wins) so the terminal outcome carries WHY it failed;
                # the snapshot exposes it to the persist callback, which decides the
                # agent's terminal state. RunPhase is left untouched (a StreamError is
                # a normal terminal bus event that clients already read as the end).
                if isinstance(event, StreamError):
                    run.error = event.detail
                run.bus.publish(event)
        finally:
            aclose = getattr(agen, "aclose", None)
            if aclose is not None:
                await aclose()
