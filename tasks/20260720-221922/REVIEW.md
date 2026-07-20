# Review: A0 agent runtime foundation (supervisor + event bus)

- TASK: 20260720-221922
- BRANCH: refactor/agent-runtime

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: out-of-context

Check suite (reviewer ran it in the worktree): ruff clean, mypy clean (31
files), `python -m pytest` = 216 passed. Verified independently in-session.

- [x] R1.1 (MAJOR) scufris/supervisor.py `start`/`_execute` - the DoD "reset waits
  for no active run of that agent" does not hold. `start()` only schedules the
  task and returns; the "chat" serialize lock is acquired later, inside
  `_execute` when the task first runs. A `reset`/`new`/`switch`/`fork`/`delete`
  (all `async with chat_lock`) arriving in that window finds the lock free and
  runs `agent.reset()` before the turn acquires it, resetting the agent mid-turn.
  The old request-held `chat_lock` closed this gap. Fix: reserve the serialize
  slot SYNCHRONOUSLY in `start()` (before returning) so a later mutation queues
  behind the turn; add a test that fires a mutation right after starting a turn
  and asserts ordering.
  - Response: Fixed. Replaced the per-key `asyncio.Lock` with a synchronous FIFO
    reservation (`reserve(key)` appends to a per-key chain and returns
    `(predecessor_future, release)`); `start()` reserves the key synchronously
    before returning, and both `_execute` and the new `serialized(key)` context
    manager (used by the mutation endpoints) await their predecessor. So a
    mutation that reserves after the turn's `start()` always waits for the turn.
    Test: `test_serialized_waits_for_an_inflight_run` (mutation reserved
    immediately after `start()`, before the turn task has drained, still runs
    last).
- [x] R1.2 (MAJOR) scufris/supervisor.py - `self._runs` grows without bound; every
  `/api/chat/stream` mints a fresh run_id and leaves a `_Run` (each owning a
  256-event `EventBus` deque) resident forever - an unbounded leak on a
  long-lived server. Fix: reap terminal runs (keep the last N done/error).
  - Response: Fixed. Terminal runs are appended to a bounded `deque`; when it
    exceeds `max_history` (default 200) the oldest terminal run is dropped from
    `self._runs`. Running/queued runs are never reaped. Test:
    `test_terminal_runs_are_reaped`.
- [x] R1.3 (MINOR) tasks/.../TASK.md DoD / tests/test_supervisor.py - the "SSE
  client disconnect does not cancel the run" guarantee is proven at the
  Supervisor primitive, not the `/api/chat/stream` endpoint (TestClient can't
  half-consume a stream). Defensible substitution; note it or add an
  endpoint-level test.
  - Response: Strengthened. Exposed `app.state.supervisor` and added
    `test_chat_stream_runs_as_a_supervised_background_job`, which drives the real
    endpoint and asserts the turn is a supervised run that reaches `done` -
    proving the run lives in the supervisor (decoupled from the request), not
    just in the primitive. A true mid-stream HTTP disconnect needs a real socket
    (heavy, per the `test-streaming-over-a-real-socket-not-asgitransport`
    lesson); the supervisor-level `test_run_survives_subscriber_disconnect`
    remains the disconnect proxy.
- [x] R1.4 (NIT) scufris/supervisor.py - `AgentRunStalled(RuntimeError)`; if a
  `make_stream` raised it directly it would be misclassified as a heartbeat
  stall. Harmless today.
  - Response: Made `AgentRunStalled` subclass `Exception` directly and documented
    that it is internal-only (raised solely by the heartbeat guard).

### Reviewer-confirmed correct (no change)
- EventBus fan-out / replay / drop-oldest / `_CLOSE` delivery / replay-vs-live
  dedup: correct.
- Lock-before-semaphore ordering (no same-key starvation of other agents):
  correct (preserved by the reservation-before-semaphore order in the fix).
- Heartbeat vs budget `TimeoutError` disambiguation: cleanly separated.
- `/api/chat/stream` regressions (headers, padding, image write/cleanup, error
  frames, disabled->503): none; the four pre-existing chat-stream tests pass
  unchanged through the bus relay.
- cwd seam test is meaningful (fails if the param were dropped).

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (targeted fixes to R1.1/R1.2 + test-only additions for
  R1.3/R1.4; each pinned by a discriminating test verified to exercise the fix)

Verification of each response against the new diff:
- R1.1: `reserve()` is a synchronous FIFO taken in `start()` before it returns;
  `_execute` and `serialized()` await the predecessor. `test_serialized_waits_for_an_inflight_run`
  reserves the mutation immediately after `start()` (before the turn task drains)
  and asserts `"reset" not in order` while the turn is blocked, then final order
  `[turn-start, turn-end, reset]`. Under the old lock-acquired-in-task behavior the
  mutation could acquire during the turn's wait, so the mid-run assertion fails -
  the test discriminates the fix.
- R1.2: `_retire` bounds `_terminal` to `max_history`, reaping oldest terminal
  runs from `_runs`. `test_terminal_runs_are_reaped` (cap 3, 5 runs) asserts the
  two oldest are gone and `list_runs()` == 3. Running/queued runs are never in
  `_terminal`, so they are never reaped.
- R1.3: `app.state.supervisor` exposed; `test_chat_stream_runs_as_a_supervised_background_job`
  drives the real endpoint and asserts the run is tracked and `done`.
- R1.4: `AgentRunStalled` now subclasses `Exception`, documented internal-only.

Suite after fixes: ruff + mypy clean, 219 passed (+3 vs round 1). No new findings.
