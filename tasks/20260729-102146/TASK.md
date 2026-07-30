# Spike: choose the transactional state persistence architecture

- STATUS: OPEN
- PRIORITY: 85
- TAGS: spike,v0.2.0,reliability,storage
- KIND: SPIKE
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

## Story

As a maintainer, I want one durable persistence design for all mutable Scufris
state, so that concurrent agents, request threads, process crashes, and schema
changes have explicit correctness guarantees rather than store-specific JSON
behavior.

## Steps

- [ ] Inventory every mutable state store after 20260729-124655 lands, including
      projects, agents, session ownership, outcomes, settings, reasoning,
      authentication sessions, host proposals/approvals, schedules, digest
      history, and any other app-owned host state. Record the root-owned
      `scufris-hostd` audit as an intentional external boundary.
- [ ] Reproduce and quantify the fixed-temporary-file race and identify other
      lost-update or partial-write windows.
- [ ] Compare SQLite transactions with locked atomic JSON snapshots, including
      async/thread interaction, migrations, backups, observability, and test
      isolation.
- [ ] Evaluate both candidates against the known next workload: ordered
      append-only conversation/activity events, stable correlation IDs,
      pagination/retention, idempotent web/Telegram delivery, and an atomic
      state-change-plus-event/outbox commit.
- [ ] Choose the state boundary and transaction model, including whether all
      stores share one database and how multi-record updates are committed.
- [ ] Define idempotent migration, rollback, backup, corruption recovery, and
      downgrade behavior for every existing JSON store, including partial
      migration and a clear policy for external/root-owned state.
- [ ] Write `SPIKE.md` with the evidence and `DECISION.md` with the selected
      architecture, rejected alternatives, and constraints.
- [ ] Refine child task 20260729-102147 if the chosen architecture changes its
      implementation or verification plan.

## Definition of Done

- The spike accounts for projects, agents, sessions, outcomes, settings,
  reasoning, authentication, host proposals/schedules, future plugin state,
  and the root-owned helper audit boundary
  (cmd: `rg -n "projects|agents|sessions|outcomes|settings|reasoning|auth|host|schedule|plugin|root-owned" tasks/20260729-102146/SPIKE.md`).
- The selected design supports ordered append-only events, stable correlation,
  idempotent delivery, bounded retention, and atomic domain-state plus event
  commits without creating the future conversation schema in this task
  (cmd: `rg -n "append-only|correlation|idempotent|retention|outbox|atomic" tasks/20260729-102146/SPIKE.md`).
- A load-bearing persistence choice and migration policy are recorded
  (cmd: `test -f tasks/20260729-102146/SPIKE.md && test -f tasks/20260729-102146/DECISION.md && tatr check 20260729-102146`).
- The original concurrent-create failure is captured as a reproducible test
  target for the implementation task
  (cmd: `rg -n "concurrent|restart|migration" tasks/20260729-102146/SPIKE.md`).
- The user accepts the selected durability and migration tradeoffs (manual: user check).

## Notes

- Epic: 20260729-102145.
- Begins after: 20260729-124655, so the inventory describes the state that
  actually exists after the host epic rather than today's smaller snapshot.
- Relevant code: `scufris/projects.py`, `scufris/agent_store.py`,
  `scufris/settings_store.py`, `scufris/reasoning_store.py`, `scufris/auth.py`,
  and the host state modules added by 20260729-124655.
- Known downstream workload: 20260729-220835. This spike chooses a store that
  can support its likely event/query shape, but does not pre-create speculative
  conversation tables.
- This task decides the mechanism. It does not migrate production state.
