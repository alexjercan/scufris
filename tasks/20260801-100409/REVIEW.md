# Review: Migrate agent, session, outcome, settings, and reasoning state

- TASK: 20260801-100409
- BRANCH: fix/migrate-agent-session-outcome-settings-reasoning-state

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (BLOCKER) scufris/app.py:2266 - making `_launch_agent_turn` a
  coroutine split the one-run-per-agent guard's check-then-register window: the
  409 check reads `supervisor.status(agent_runs[id])`, which is None until
  `supervisor.start` registers the run at app.py:2400, and `await
  asyncio.to_thread(agents.mark_running, ...)` at app.py:2398 now yields between
  the two. Two concurrent turns for one agent both pass the check, and
  `agent_runs[agent.id]` is overwritten so the earlier run can no longer be
  stopped - contradicting `stop_agent_run`'s "at most one run is live per agent -
  the single `agent_runs[agent_id]` entry is the one to stop" (app.py:2948).
  Measured: 8 simultaneous `POST /api/agents/builder/chat` return
  `[200, 200, 409 x6]` on this branch and `[200, 409 x7]` on master.
  Register the run synchronously before the first await - either mark the slot in
  `agent_runs` and have the guard test membership rather than
  `supervisor.status`, or call `supervisor.start` before offloading
  `mark_running` - and pin it with a test that fires the second turn WITHOUT
  first polling `/status` (the existing
  `test_agent_chat_conflicts_with_active_run` waits for the run to register, so
  it never enters the window).
  Response: fixed. `launching_runs` holds a run id from the moment
  `_launch_agent_turn` claims `agent_runs[id]` until `supervisor.start` takes
  over, and `_agent_run_active` - now the ONE predicate the guard, the wake
  bridge's is-busy check and fork all use - reads it alongside the supervisor.
  The claim happens in the same synchronous step as the check. Pinned by
  `test_simultaneous_agent_chats_start_exactly_one_run`, which fires 8 chats
  with NO `/status` poll and slows `mark_running` by 50ms so the window is
  entered every run rather than won by luck (5/5 red with the reservation
  check removed, where the unslowed version was red only 3/5). DECISION.md 6.

- [x] R1.2 (MAJOR) tests/test_app.py:3989 -
  `test_agent_routes_do_not_stall_the_event_loop` is the DoD proof for "the
  routes stay responsive while another writer holds the write lock", and it
  cannot fail on that criterion. `release.set()` runs BEFORE the four route
  calls, so the write lock is no longer held while they are driven; and the
  assertion `ticks > hold / 0.02` (25) is satisfied by the preceding `await
  asyncio.sleep(0.5)` alone. Instrumented on this branch: 52 total ticks, of
  which only 3 fall in the route window - the routes could stall the loop
  completely and the test would still pass. Move `release.set()` after
  `statuses`, bound the holder's own wait so it cannot deadlock against the
  routes, and assert on the ticks counted only across the route window.
  Response: fixed. `release.set()` moved after `statuses`; the holder's own
  wait is bounded by `hold` so it cannot deadlock against the four routes that
  all block on the same lock; ticks are sampled before/after the route window
  and asserted against the measured window, plus a `window >= hold * 0.8`
  assertion that the routes really did wait. Sabotage: a `time.sleep(0.5)` on
  the loop inside `GET /api/agents` now fails it (4 ticks against 13 required).

- [x] R1.3 (MAJOR) scufris/app.py:3577 - the comment "The set-then-launch is
  synchronous, so nothing interleaves" is now false and it is load-bearing: it is
  the stated reason `fork_session` takes no
  `supervisor.serialized(ORCHESTRATOR_ID)` lock. There are three awaits between
  `await asyncio.to_thread(agents.set_orchestrator_session, None)` (app.py:3578)
  and `await _launch_agent_turn(...)` (app.py:3580), so a concurrent orchestrator
  turn can land on the just-cleared session and start a new chat instead of
  resuming the operator's. Close the window (with R1.1's fix, or by holding the
  reservation across the sequence) and rewrite the comment to say what actually
  holds.
  Response: fixed. The forked record is derived in memory
  (`orchestrator.model_copy(update={"session_id": None})`) instead of re-read,
  so no await sits between the clear and the launch's claim; with R1.1 that
  claim is synchronous, which makes clear-then-launch atomic again. Comment
  rewritten to state that, and why the record is not re-read.

- [x] R1.4 (MINOR) scufris/wake.py:17 - the module docstring still claims the
  completion callback "runs to completion before another completion starts it
  again ... the pending map needs no lock". `on_run_complete` now awaits
  `asyncio.to_thread(self._agents.outcome, ...)` (wake.py:99) and `await
  self._drain()`, and two runs finishing concurrently are two separate `_execute`
  tasks, so they DO interleave: an entry re-added to `self._pending` during
  another completion's `await self._launch(...)` is popped by that completion's
  `batch` loop (wake.py:107) and its wake is dropped. Either re-derive the claim
  and correct the docstring, or key the pop on the batch's own values.
  Response: fixed. Both. The pop is now conditional on `self._pending.get(id)`
  still equalling the batch's own value, so a wake re-recorded mid-launch
  survives; the docstring drops the runs-to-completion claim and states the two
  interleavings that DO happen and why neither costs a wake (the second - two
  drains both passing `_is_busy` - was already handled by the 409 branch).
  `test_wake_recorded_during_a_launch_is_not_dropped` pins it and is red with the
  unconditional pop restored.

- [x] R1.5 (MINOR) scufris/db/legacy.py:233 and scufris/README.md:530 - "a
  damaged `outcomes.json` does not hold back the agents ... it refuses that ONE
  source, names it, and the rest come in" is only true of sources imported
  EARLIER. `import_agent_state` (legacy.py:228-243) calls the five imports with
  no per-source `try`, and the refusal propagates out of `open_state_database`,
  so a bad `sessions.json` stops all four later sources and the server does not
  start at all. Either wrap each source so a refusal is collected and re-raised
  after the rest have run, or state the ordering truth in both places.
  Response: fixed, both halves. `import_agent_state` builds the five sources as a
  list and runs each in its own `try`, collecting `LegacyImportRefused` and
  raising them joined at the end - so startup still fails (a silently skipped
  source would be the tolerant loader this package refuses to be) but the intact
  sources are in with their gate rows and the retry re-reads only the damaged
  file. The claim in the docstring and in scufris/README.md now says exactly
  that, including why running the later sources anyway is safe: sessions-before-
  agents degrades correctly, since with no mappings imported the record's own
  `session_id` IS the right answer.
  `test_a_damaged_source_does_not_hold_back_the_sources_after_it` damages the
  FIRST source, which is what tells this apart from "everything before the damage
  got in".

- [x] R1.6 (MINOR) scufris/checks.py:135 and scufris/hostd/protocol.py:189 - both
  files are reformatted (line wrapping only) with no change this task needs, and
  both already pass `ruff check` on master. This is the churn the Steps
  explicitly warned about ("the same sweep in 20260801-120412 reformatted five
  unrelated files"). Revert both files to their master content.
  Response: fixed. Both reverted with `git checkout master --`; the branch no
  longer touches either file.

Verified by this reviewer:

- `ruff check . && mypy . && python -m pytest` - green (exit 0, 185 source files,
  full suite passed).
- All six named `test:` proofs pass by name; the `! rg ... json.tmp` proof
  returns no matches.
- Re-derived R1.1 independently by driving 8 simultaneous chat POSTs against a
  built app on this branch and on master.
- Re-derived R1.2 independently by instrumenting the tick counter around the
  route window.
- Legacy import: read the five loaders, the gate-key collision test, and the
  ten-case damage parametrization; the fixtures under
  `tests/fixtures/legacy_state` cover all five shapes and the idempotence half
  re-runs the import.
- Doc sweep: no stale `agents.json` / `sessions.json` / `outcomes.json` /
  `settings.json` mention survives outside `tasks/`, the CHANGELOG history, and
  the legacy importer that is about those files by name.
- DECISION.md covers all five load-bearing choices; the split of
  `agent_store/store.py` into `reserved`/`signals`/`rows` keeps every file under
  the 600-line cap (largest 476).

Not verified: no `manual:` proofs are open on this task.

Process signal: `_launch_agent_turn` becoming a coroutine (DECISION.md 2) turned
every previously-atomic sync sequence in `create_app` into an interleavable one.
The Steps swept the call sites for "who calls this, from which thread" but not
for "which check-then-act sequence did the new await just split" - R1.1, R1.3 and
R1.4 are three instances of that one omission.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [ ] R2.1 (MINOR) scufris/app.py:3616 - the rewritten comment says the clear
  and the launch's slot claim "are ONE synchronous step ... there is no await
  between it and the clear", but the clear IS an await
  (`await asyncio.to_thread(agents.set_orchestrator_session, None)`,
  app.py:3624) and resuming from it is a scheduling point: the loop is free to
  run another task between the store write committing in the worker thread and
  this task reaching `_launch_agent_turn`'s claim. The window is one loop turn
  with no I/O in it, and the residual harm is bounded and visible - a wake drain
  that wins the claim there starts the fresh orchestrator chat and the operator's
  fork 409s, rather than the round-1 harm of the fork silently losing its seed -
  so this is accuracy, not a live defect. Either claim the slot before the clear,
  or say "no await between the clear RETURNING and the claim, so only a
  same-tick reschedule fits, and the loser of it gets a 409" instead of denying
  the interleaving outright.
  - Response:

Verified by this reviewer:

- `ruff check . && mypy . && python -m pytest` - green, exit 0.
- All six R1 fixes re-derived by sabotage on this branch, each restored after:
  deleting the `run_id in launching_runs` branch makes
  `test_simultaneous_agent_chats_start_exactly_one_run` fail 3/3 (R1.1);
  restoring the unconditional `self._pending.pop` fails
  `test_wake_recorded_during_a_launch_is_not_dropped` (R1.4); dropping the
  per-source `try` fails
  `test_a_damaged_source_does_not_hold_back_the_sources_after_it` (R1.5).
- R1.2 independently: a pure loop stall (0.5s blocking sleep in an `async def`
  `GET /api/agents` that still offloads its store call) now fails the proof with
  "only 4 heartbeats in 0.541s" against 13 required - the implementer's reported
  numbers reproduce. A stall that FastAPI already offloads (the same sleep in the
  `def` route) correctly still passes. Note that the naive sabotage - store call
  moved onto the loop - is caught earlier by the task's own `transaction()`
  guard, which is its own evidence that the guard works.
- R1.6: `git diff master -- scufris/checks.py scufris/hostd/protocol.py` is
  empty; the branch no longer touches either file.
- R1.3 read rather than driven: the record is derived in memory and the re-read
  is gone, so the three-await gap the finding named is closed. R2.1 is what is
  left of it.
- `! rg -n 'with_suffix\("\.json\.tmp"\)' ...` returns no matches (exit 1).
- DECISION.md 6 records the claimed-run set, including the rejected alternative
  (start before offloading `mark_running`) and why. All Steps in TASK.md ticked.

Not verified: no `manual:` proofs are open on this task.
