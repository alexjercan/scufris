# Decision: Migrate agent, session, outcome, settings, and reasoning state

- DATE: 20260802-203934
- STATUS: ACCEPTED
- TASK: 20260801-100409
- TAGS: storage,concurrency,agents

## Context

20260801-120412 moved ONE store (projects) onto `Database.transaction()`. Its
review found the failure mode that generalizes to every remaining cutover: three
`async def` routes reached the store directly, so a `BEGIN IMMEDIATE` ran on the
event loop thread and held SQLite's single write lock there - measured at a 3.04s
loop stall against a 0.01s heartbeat, and a 500 past `busy_timeout`. The stores
this task moves are reached from far more loop-thread code than projects were:
20 `agents.*` / `reasoning_store.*` call sites inside `async def` bodies in
`app.py`, plus the run engine's `persist` callback, which the supervisor invokes
from inside `_execute`'s `finally` - on the loop.

Two of those call sites cannot simply be offloaded. `persist` must stay ordered
with respect to the wake bridge and the deferred-decision drain that follow it,
and those launch turns (`supervisor.start` -> `asyncio.create_task`), so they
need the loop. And `CodexBackend.read_transcript` constructs a `ReasoningStore`
from `settings` alone: the `AgentBackend` protocol passes no handles, and it is
the seam whose whole point is that nothing above it branches on backend.

## Decision

**1. `Database.transaction()` refuses to open on a thread with a running event
loop.** The rule `engine.py` already states in prose becomes a `RuntimeError`
naming `asyncio.to_thread`. A `def` FastAPI route (threadpool) and a plain
synchronous caller both have no running loop and are unaffected; an `async def`
caller gets a loud error at the first call instead of a lock held under the
loop. The remaining epic task migrates four more stores into exactly this
hazard, so the guard is worth more than the one sweep it forces here.

**2. The supervisor's `on_complete` may be a coroutine function, awaited in
place.** `persist` becomes `async`: it offloads `mark_finished` with
`asyncio.to_thread` and then runs its loop-bound tail (`wake_bridge`,
`_drain_deferred_decision`) in the same `finally`, so the ordering the current
code documents is preserved. `_launch_agent_turn` becomes `async` for the same
reason - it calls `mark_running` and then `supervisor.start`.

**3. One `Database` handle per process, reached through a shared accessor.**
`mcp_stores.database(settings)` already memoizes one handle per resolved state
dir; that memo is promoted to a module both `create_app` and the leaves use, and
`create_app`'s lifespan closes and evicts it. `CodexBackend.read_transcript`
reaches the reasoning store through it. The epic's "ONE boundary per process"
premise is kept: there is still exactly one handle, it is just reachable by name
as well as by injection.

**4. Session history is ordered rows, not a JSON column.**
`agent_session_history(agent_id, seq)` with `agent_session(agent_id)` holding the
current pointer, backend and spawn parent. `add`, `set_current` and `remove` are
then row operations inside the transaction rather than a read-modify-write of a
list.

**5. Reasoning turns are `(session_id, seq)` rows, and a failed append raises.**
20260801-100405 measured the per-session snapshot rewrite growing with history;
rows make an append O(1). The swallow at `reasoning_store.py:120` is deleted
rather than ported - it is why 186 of 200 turns disappeared with no failed
request in 20260729-102146.

**6. A claimed-run set backs the one-run-per-agent guard (review round 1).**
Decision 2 turned `_launch_agent_turn` into a coroutine, and that split a
check-then-act the whole file relied on: the 409 guard asked
`supervisor.status(agent_runs[id])`, which answers None until `supervisor.start`
registers the run, and the `mark_running` offload now yields in between. Measured:
8 simultaneous chats for one agent returned two 200s, and only the last writer of
`agent_runs[id]` is stoppable by `cancel_agent_run`. `launching_runs` holds run
ids claimed but not yet started; `_agent_run_active` reads it alongside the
supervisor and is the single predicate the guard, the wake bridge's is-busy check
and `fork_session`'s atomicity all rest on. Claiming the slot in the same
synchronous step as the check is what restores the pre-coroutine invariant -
starting the run before offloading `mark_running` would also close the window,
but it lets a fast turn record DONE before RUNNING is written.

## Alternatives considered

- **Sweep the loop-thread call sites without the guard.** How it would work:
  find the 20 sites, wrap each in `asyncio.to_thread`, rely on review for the
  next four stores. Rejected: this is the second time the same defect has been
  found by measurement after the fact, and the sweep's own completeness is
  unprovable - a proof that "no site was missed" is a proof about the boundary,
  not about a list.
- **A latency test instead of the guard.** Hold the write lock from another
  thread, drive each async route, assert a heartbeat keeps ticking. Rejected as
  the ONLY mechanism: it proves the routes that the test enumerates, which is
  the same list problem, and it is slow. Kept as one proof, not as the design.
- **Passing a `Database` (or a `ReasoningStore`) through
  `AgentBackend.read_transcript`.** Rejected: four adapters and six call sites
  change so that one adapter can read one sidecar, and the handle would then be
  in a protocol whose other three implementations have no use for it.
- **Merging reasoning above the backend seam**, in the four `app.py` transcript
  routes where `reasoning_store` is already in scope. Rejected: the merge is
  codex-only, so the app layer would have to branch on backend name - the exact
  thing `backends/__init__.py` says nothing above the seam does.
- **Keeping the JSON stores and adding a lock.** Rejected upstream by
  20260801-100405 on measurement (multi-record commits 100/100 torn, 150 of 300
  cross-process writes lost silently); repeated here only because this task is
  the multi-record commit that measurement was about.

## Consequences

Easier: the completion path (registry update + outcome append + agent row) is
one commit, so the orchestrator can never poll an outcome whose session record
did not land. `mark_finished`'s `preserve_signal` read, `OutcomeStore.acknowledge`,
`SessionRegistry.add`/`remove` and `SettingsStore.apply` all close their
read-modify-write window at the transaction. A reasoning append stops rewriting
the whole session.

Harder: the async surface of `app.py` grows - `_launch_agent_turn` and `persist`
become coroutines, and every store call in an `async def` gets an
`asyncio.to_thread` wrapper, which is noise at each site. The guard means any
future async caller fails loudly rather than degrading, which is the intent, but
it will also fire on async TESTS that call a store directly; those move to
`to_thread` or to a sync body. `agent_store/store.py` is 5 lines under the
600-line cap today and must be split as part of this change. And the process-wide
accessor is a second way to reach the handle: a caller that could be injected and
uses the accessor instead is a review finding, not a compile error.
