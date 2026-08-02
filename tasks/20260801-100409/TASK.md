# Migrate agent, session, outcome, settings, and reasoning state

- STATUS: CLOSED
- PRIORITY: 79
- TAGS: bug, v0.2.0, reliability, storage, agents
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-120412

## Story

As a Scufris operator, I want agent, session, outcome, settings, and reasoning
state on the transactional core, so that simultaneous agent completions never
drop a session record or an outcome that the UI already reported.

## Steps

- [x] Write the five proofs first and watch each fail on the base commit:
      simultaneous completion callbacks keeping every record across an app
      reconstruction; a completion whose outcome write fails leaving NO
      partial agent/session record; a legacy fixture of all five JSON sources
      importing exactly once; a reasoning append that cannot be written
      raising instead of logging; and a `Database.transaction()` opened on a
      thread with a running event loop raising.
- [x] Declare the schema in `scufris/db/models.py` and autogenerate ONE Alembic
      revision for it (the drift test `test_schema_has_no_pending_autogenerate_diff`
      is the check that the revision matches):
      `agents` (every `AgentRecord` field except `session_id`, `id` PRIMARY KEY),
      `agent_session` (`agent_id` PK, `backend`, `current_session_id` nullable,
      `parent_agent_id` nullable, `parent_session_id` nullable),
      `agent_session_history` (`agent_id`, `seq` composite PK, `session_id`) so
      switcher order is stored rather than reconstructed,
      `agent_outcome` (`agent_id` PK, `state`, `message`, `run_id`,
      `session_id` nullable, `ts`, `acknowledged`),
      `settings_override` (`key` PK, `value` JSON text),
      `reasoning_turn` (`session_id`, `seq` composite PK, `answer`, `reasoning`).
- [x] Add the loop-thread guard to `scufris/db/engine.py` (DECISION.md 1):
      `transaction()` raises `RuntimeError` when `asyncio.get_running_loop()`
      succeeds, with a message naming `asyncio.to_thread`. Land this BEFORE the
      store rewrites so the sweep below is driven by failures, not by reading.
- [x] Sweep the loop-thread callers this makes illegal. The 20 known sites are
      `app.py` lines 877, 915, 2280, 2301, 2797, 2885, 3022, 3498, 3500, 3501,
      3512, 3521, 3522, 3526, 3561, 3563, 3565, 3605, 3661, 3679; re-derive the
      list with the AST sweep rather than trusting these numbers after the file
      moves. Plus the two the supervisor drives: `persist` (`app.py:2339`,
      invoked from `Supervisor._execute`'s `finally`) and `mark_running`
      (`app.py:2367`). Per DECISION.md 2: `Supervisor.start` accepts a coroutine
      `on_complete` and `_execute` awaits it when it returns an awaitable;
      `persist` and `_launch_agent_turn` become `async`, offloading only the
      store call with `asyncio.to_thread` and keeping the loop-bound tail
      (`wake_bridge.on_run_complete`, `_drain_deferred_decision`,
      `supervisor.start`) on the loop and in its current order. Confirm the
      caller chain of `_launch_agent_turn` (`app.py:2438`, `2519`, `2886`) and
      of `_drain_deferred_decision` before changing signatures: whichever sync
      callers remain become `async` with it, or the change stops there.
- [x] Promote the one-handle-per-process memo out of `scufris/mcp_stores.py`
      (DECISION.md 3) so `create_app` and the leaves share ONE `Database`:
      `create_app` takes its handle from the accessor and the lifespan closes
      AND evicts it, and `CodexBackend.read_transcript` (`backends/codex.py:110`)
      builds its `ReasoningStore` on it. Verify with the existing MCP
      cross-process proof that the subprocess path still opens its own.
- [x] Rewrite `scufris/agent_store/registry.py` and `outcomes.py` as
      row-backed stores that take an OPEN `Connection`, and give each a thin
      `Database`-owning wrapper for the single-record calls. Nesting is an
      error on this engine, so passing the connection down is what lets the
      completion path be one transaction.
- [x] Rewrite `scufris/agent_store/store.py` onto the core: delete `_agents`,
      `_load` and `_persist`; every method opens one transaction and reads
      through. Keep the public API, `AgentsReadOnly` / `InvalidAgent` /
      `ReservedAgent` / `AgentNotFound`, the reserved-agent synthesis, the
      in-memory `_orch_state` / `_host_state` (neither has a row), `list`
      ordering by lowercased name with the host record prepended, and the
      `_with_session` read-time attach. Return `AgentRecord`, never a row.
      Move `_unique_id` inside the insert's transaction and map the PRIMARY KEY
      violation to a domain error rather than letting `IntegrityError` reach a
      route.
- [x] Split `agent_store/store.py` while rewriting it: it is 595 lines against
      the 600-line `SOURCE_CAP` and the ALLOWLIST is a ratchet that must not
      grow. The reserved-agent and signal-mutator halves are the seam.
- [x] Close the read-modify-write windows INSIDE the transaction, not around
      the write: `mark_finished`'s `preserve_signal` read of the existing
      outcome, `OutcomeStore.acknowledge`, `SessionRegistry.add`,
      `set_current` and `remove`, and `SettingsStore.apply`'s override
      read-modify-write.
- [x] Commit the whole completion path in ONE transaction: the agent row's
      terminal state, the registry's session add, and the outcome append, so an
      outcome the orchestrator can poll never exists without the session record
      it names.
- [x] Rewrite `scufris/settings_store.py` onto `settings_override` rows,
      keeping `WRITABLE_KEYS`, `REBUILD_KEYS`, the in-place live-`Settings`
      mutation, the rollback on `ValidationError`, and the `on_change` contract.
      Keep the `_apply_overrides(drop_invalid=True)` tolerance at LOAD - a
      stale key must still not crash the server on boot - and record why it
      differs from the importer's strictness.
- [x] Rewrite `scufris/reasoning_store.py` onto `reasoning_turn` rows: an
      append is one insert at the next `seq`, `read` selects by `seq`, and the
      `OSError` swallow at `reasoning_store.py:120` is DELETED. Keep
      `_SAFE_SESSION_ID` (a bad id stays a no-op) and the empty-reasoning
      1:1 alignment entry the transcript merge depends on.
- [x] Add the legacy importers to `scufris/db/legacy.py` on the existing
      `import_legacy_file` policy (backup, refuse-damaged, all-or-nothing, one
      `legacy_import` row): `agents.json`, `sessions.json`, `outcomes.json`,
      `settings.json`, and each `reasoning/<session_id>.json`. The agent loader
      applies the two migrations the deleted tolerant loader did, BEFORE
      validating, or a real operator file is refused: `write_enabled` ->
      `permission_mode`, and `canonical_backend` on the stored backend. It also
      moves a pre-registry record `session_id` into the session tables, with an
      existing mapping winning. The registry loader accepts the legacy
      `{backend, session_id}` shape as a one-element history.
- [x] Give the reasoning files an explicit gate key rather than `source.name`:
      a session id of `sessions` yields `sessions.json` and would collide with
      the registry's own `legacy_import` row. Add the key parameter to
      `import_legacy_file` and pass `reasoning/<name>`.
- [x] Wire the new importers into `scufris.db.open_state_database` next to
      `import_projects`, so they run after the migration and before any store
      read.
- [x] Add duplicate-import and corrupt-input tests per source in
      `tests/test_db_legacy.py`, and a fixture directory under `tests/fixtures`
      holding one populated example of each of the five legacy shapes.
- [x] Update the ~75 store constructions across `tests/test_agent_store.py`,
      `test_agent_outcomes.py`, `test_agent_sessions.py`, `test_settings_store.py`,
      `test_reasoning_store.py`, `test_wake.py`, `test_app.py`,
      `test_mcp_server.py`, `test_backends.py` and `test_host_agent.py` onto the
      `database` fixture. Re-read the diff rather than trusting a bulk edit: the
      same sweep in 20260801-120412 reformatted five unrelated files.
- [x] Update `scufris/README.md` section 9 (agents, sessions, outcomes,
      settings and the reasoning sidecar are no longer "one JSON file each until
      their own cutover tasks") and section 8's module map where it names the
      JSON files, and add a `CHANGELOG.md` entry.

## Definition of Done

- Simultaneous completions lose no session, outcome, or reasoning record
  (test: `test_concurrent_agent_completions_persist_every_record`).
- The completion path is atomic across its records
  (test: `test_agent_completion_commits_as_one_transaction`).
- Legacy agent-state JSON fixtures import exactly once and preserve every
  supported field (test: `test_legacy_agent_state_migrates_idempotently`).
- Reasoning turns are rows, not per-session files, and a turn is never lost
  silently (test: `test_reasoning_turns_persist_without_swallowing_errors`).
- A transaction opened from the event loop thread raises instead of holding the
  write lock there (test: `test_transaction_refuses_the_event_loop_thread`).
- The agent, session and settings routes serve records that round-trip through
  the database, and stay responsive while another writer holds the write lock
  (test: `test_agent_routes_do_not_stall_the_event_loop`).
- No agent-state store uses the fixed shared temporary-file write
  (cmd: `! rg -n 'with_suffix\("\.json\.tmp"\)' scufris/agent_store/ scufris/settings_store.py scufris/reasoning_store.py`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Review round 1 fixes

What and why. All six findings addressed on the branch. Three of them (R1.1,
R1.3, R1.4) are one defect wearing three hats: making `_launch_agent_turn` a
coroutine (DECISION.md 2) inserted an await inside sequences that used to be
atomic by construction, and each of those sequences had a comment asserting the
atomicity it had just lost. The fix restores the invariant at its source - a
`launching_runs` claim set plus one `_agent_run_active` predicate (DECISION.md 6)
- rather than patching each caller: `fork_session` then only had to stop
re-reading the record it already held, and the wake bridge only had to stop
assuming its drain runs alone.

Alternatives. For R1.1, calling `supervisor.start` before offloading
`mark_running` also closes the window and needs no new state, but it lets a fast
turn write its terminal DONE before RUNNING lands, which parks the agent as
RUNNING forever. For R1.5, the reviewer offered "state the ordering truth in the
docs" as an alternative to per-source isolation; isolation was chosen because the
policy the module is built on is per-source, and the doc was describing the
design rather than the code.

Difficulties and diagnosis. The first version of the R1.1 proof reproduced the
leak only 3 runs in 5: the eight requests each make two store round trips before
the guard, so the winner usually registers before the second arrives. Slowing
`mark_running` by 50ms in the test makes entering the window deterministic (5/5
red under sabotage) without changing what is asserted - the defect is that the
window exists, not that it is wide.

Evidence. `ruff check . && mypy . && python -m pytest` green (956 passed, exit
0); all six named `test:` proofs pass by name; the `! rg ... json.tmp` proof
returns no matches. Each new test was verified red against its own defect:
R1.1's by neutralizing the `launching_runs` check (5/5), R1.2's by a
`time.sleep(0.5)` on the loop inside `GET /api/agents` (4 ticks against 13
required), R1.4's by restoring the unconditional pop.

Reflection. The review's process signal is right and is worth carrying: a sweep
for "who calls this, from which thread" does not find "which check-then-act did
the new await just split". The second sweep is cheap to run - the awaits are all
new in the diff - and its evidence is a comment that says "synchronous" or
"cannot interleave" next to one.

## Notes

- Epic: 20260729-102145. Lane B, fifth of six.
- DECISION.md records the five load-bearing choices: the loop-thread guard on
  `Database.transaction()`, the awaitable `on_complete`, one process-wide
  handle reached by accessor, session history as ordered rows, and reasoning
  turns as rows with the swallow deleted.
- Depends on the persistence core task; that task owns the transaction API.
- Auth, host, schedule, and digest state migrate in the successor task
  (20260801-100413); they keep working on JSON until then.
- The reasoning sidecar's error swallowing (`reasoning_store.py:120`) is
  removed, not ported: it is why 186 of 200 turns disappeared with no failed
  request in 20260729-102146.
- Breadth, recorded rather than split: this is five stores, six tables, an
  importer per source, and a sweep across `app.py` and `supervisor.py`. The
  clean split is settings + reasoning as their own task, since neither shares a
  transaction with the completion path - but the loop-thread guard has to land
  with the agent trio, and the reasoning append is one of the loop-thread call
  sites, so the second task would inherit a rule its own work must satisfy
  anyway. The Steps are ordered so the settings and reasoning rewrites are the
  last two commits before the legacy import, which is where a split would cut.
- 20260801-120412's review lesson applies directly: moving a store onto the
  database changes every caller's cost model, so the sweep is "who calls this,
  from which thread", not "who constructs this".
