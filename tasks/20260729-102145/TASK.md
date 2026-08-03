# EPIC: Make Scufris durable and backend-truthful

- PRIORITY: 110
- TAGS: goal, epic, v0.2.0, reliability, backend
- KIND: EPIC
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Epic

Make Scufris safe under the concurrent writes produced by multiple agents and
make every operator surface report the backend that is actually running.
This epic addresses the two highest-risk findings from the 2026-07-29 project
audit before broader automation work builds on the current state layer.

## Done Means

1. Concurrent API mutations and simultaneous agent completions persist without
   exceptions, lost records, or corrupt state across a restart
   (test: `test_concurrent_state_mutations_survive_restart`).
2. Landing, agent settings, and Telegram report the same effective backend,
   model, health, tools, memory, and quota semantics for Codex, Claude,
   OpenCode, and mock agents
   (test: `test_orchestrator_surfaces_are_backend_consistent`).
3. The migration and recovery behavior is documented for an existing Scufris
   state directory (cmd: `rg -n "migration|backup|recovery" README.md scufris/`).
4. All app-owned mutable state present after the host-operator epic, including
   authentication and host proposal/schedule state, shares the selected
   transactional and recovery boundary; the root-owned privileged audit remains
   an explicit external boundary
   (test: `test_post_host_state_uses_declared_persistence_boundary`).
5. manual: with two agents completing and a host proposal changing state at the
   same time, their sessions, outcomes, and proposal remain visible after
   restarting Scufris.

## Child Tasks

Four lanes, each run in dependency order. Derive live status with
`tatr frontier 20260729-102145`.

Lane A - persistence decision:

- [x] 20260729-102146 (p85, v0.2.0) inventory app-owned mutable state and
      reproduce the write races
- [x] 20260801-100405 (p84, v0.2.0) choose the persistence mechanism,
      migration, and recovery policy - SQLite, one database, recorded in its
      DECISION.md

Lane B - persistence implementation:

- [x] 20260729-102147 (p83, v0.2.0) add the SQLAlchemy transactional engine
      core
- [x] 20260801-120404 (p82, v0.2.0) land the Alembic migration runner and the
      projects schema
- [x] 20260801-120407 (p81, v0.2.0) import legacy JSON state into the database
- [x] 20260801-120412 (p80, v0.2.0) cut the project store over to the database
- [x] 20260801-100409 (p79, v0.2.0) migrate agent, session, outcome, settings,
      and reasoning state
- [x] 20260801-100413 (p78, v0.2.0) migrate auth, host, schedule, and digest
      state with a legacy JSON import path
- [x] 20260803-002141 (p70, v0.2.0) move the configuration-change registry onto
      the database - the last app-owned in-memory store, found by the boundary
      test in 20260801-100413
- [x] 20260803-014401 (p40, v0.2.0) make the config-change restart proofs reopen
      the database and cover the reap bound - review round 1 of 20260803-002141,
      MINOR
- [x] 20260803-113000 (p35, v0.2.0) prove the startup sweep clears a building
      row orphaned by a crash - seeded by 20260803-014401 DECISION.md 1, where a
      clean shutdown turned out to write `cancelled`, not `building`

Lane C - backend truth:

- [x] 20260729-102148 (p75, v0.2.0) extract the backend-aware orchestrator
      diagnostics service
- [x] 20260801-100415 (p74, v0.2.0) delegate legacy `/api/agent/*` routes to
      orchestrator diagnostics
- [x] 20260801-100419 (p73, v0.2.0) align Telegram and the UI with
      orchestrator diagnostics
- [x] 20260803-032950 (p60, v0.2.0) make the health session count follow the
      orchestrator backend - the last codex-shaped read on the health surface,
      found while delegating the legacy routes
- [x] 20260803-034922 (p45, v0.2.0) pin the two legacy-diagnostics tests that
      cannot go red - review round 2 of 20260801-100415, MINOR and NIT
- [x] 20260803-042958 (p40, v0.2.0) clear the round-1 MINOR findings from the
      diagnostics alignment

Lane D - assembly refactor:

- [x] 20260801-100425 (p72, v0.2.0) characterize app routes and extract the
      auth and host routers
- [x] 20260803-061210 (p42, v0.2.0) clear the round-2 findings from the router
      extraction
- [x] 20260801-100441 (p71, v0.2.0) extract the orchestrator-turn and
      agent-run services
- [x] 20260729-103712 (p70, v0.2.0) extract the remaining routers and reduce
      `create_app` to assembly
- [x] 20260803-102351 (p20, v0.2.0) close the round-2 findings from the
      `create_app` assembly extraction

## Decisions

- Recorded 20260729-102147 DECISION.md: SQLAlchemy 2.0 + Alembic replace stdlib
  `sqlite3` and the `PRAGMA user_version` ladder, superseding sections 1 and 4
  of the decision below. Everything that decision MEASURED is unchanged - one
  database, the four pragmas, `BEGIN IMMEDIATE`, the transaction as the
  read-modify-write boundary, no in-memory mirror, damaged-is-not-empty, a
  synchronous API reached through `asyncio.to_thread`, file-backed test
  fixtures, and the whole legacy-import policy. The legacy import moves off the
  version ladder onto a `legacy_import` table because it needs
  `Settings.state_dir` and pydantic validation. Paid: `greenlet` and `Mako` in
  the uv2nix closure, pragmas and `BEGIN IMMEDIATE` as event hooks. Bought:
  autogenerated reviewable revisions, a model-vs-schema drift test, and a
  declarative schema the two follow-up migrations extend. Sync engine only; the
  async one would need the `aiosqlite` that decision rejected on measurement.
- Recorded 20260801-100405 DECISION.md: one SQLite database at
  `<state_dir>/scufris.db` through stdlib `sqlite3` (WAL, `synchronous=FULL`,
  `busy_timeout`, `foreign_keys`, 0600), a connection per thread, and one
  synchronous `db.transaction()` over `BEGIN IMMEDIATE` that loop-thread
  callers reach through `asyncio.to_thread`. Migration is a `PRAGMA
  user_version` ladder; each version's legacy-JSON import is one transaction
  that backs up, validates, never deletes a legacy file, and refuses damaged
  input by name, and a store not yet at its version keeps reading its JSON. The rejected alternative is locked atomic JSON snapshots - it passes
  the single-store concurrency test and fails on multi-record commits (100/100
  torn), reader latency (91ms p50), cross-process writes (150 of 300 lost
  silently) and append-only cost. Done Means 4 is answered: host proposals join
  the boundary as a durable decision journal while the root helper stays
  authoritative for the pending set; the reasoning sidecar becomes rows; auth
  sessions join; the privileged audit stays external.
- Recorded 20260729-102146 SPIKE.md: the state inventory, mutator matrix, and
  the reproduced write race the decision argues against. Two findings bear on
  Done Means 4: host proposals have NO persisted store today
  (`host_actions.HostActionStore` is memory-only, rebuilt from the root helper),
  and auth sessions are already lock-protected. 20260801-100405 owes an explicit
  answer on whether the host proposal store joins the boundary.

## Manual Acceptance

- (accepted 2026-08-01) 20260801-100405: the durability and migration tradeoffs
  of the selected persistence architecture - SQLite over locked JSON snapshots,
  ~4x disk on append-heavy state, ~10ms per isolated test fixture, and a
  downgrade path that is one-way once the operator deletes their legacy JSON
  files.
- (accepted 2026-08-03) 20260801-100413: existing local state migrates without
  losing projects, agents, sessions, outcomes, settings, authentication state,
  or app-owned host state.
- (accepted 2026-08-03) 20260801-100419: backend/account information feels
  consistent across the landing page, agent settings, and Telegram.
- (accepted 2026-08-03) Done Means 5: two agents completing while a host
  proposal changes state leave their sessions, outcomes, and proposal visible
  after a Scufris restart.

## Downstream v0.2.0 Readiness

This epic is the persistence/service foundation for the actor-aware
orchestrator direction. It does not implement that product behavior. The
v0.2.0 readiness work scheduled after it is:

- 20260729-220835: conversation/flow architecture spike plus interactive HTML
  mockup.
- 20260729-102151 through 20260729-102154: deterministic browser QA.
- 20260729-102158: structured tatr task metadata.
- 20260729-102203: durable agent-run activity and hierarchy.
- 20260729-102205 and 20260729-102206: reusable agent preset architecture and
  implementation.
- 20260729-102202: responsive/accessibility baseline.
