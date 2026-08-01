# Spike: choose the persistence mechanism, migration, and recovery policy

- STATUS: OPEN
- PRIORITY: 84
- TAGS: spike,v0.2.0,reliability,storage
- KIND: SPIKE
- FLOW STEP: BACKLOG
- PLAN STATUS: DRAFT
- PARENT: 20260729-102145
- DEPENDS ON: 20260729-102146

## Story

As a maintainer, I want one recorded persistence mechanism with an explicit
transaction, migration, and recovery policy, so that the three implementation
tasks share a single boundary instead of each store inventing its own
durability story.

## Steps

- [ ] Read the predecessor `SPIKE.md` inventory and reproduction; carry its
      store list and mutator matrix forward as the design input.
- [ ] Compare SQLite transactions against locked atomic JSON snapshots on:
      async/thread interaction, multi-record commits, migrations, backups,
      observability, and pytest isolation.
- [ ] Evaluate both against the known next workload from 20260729-220835:
      ordered append-only activity events, stable correlation IDs, pagination
      and retention, idempotent web/Telegram delivery, and an atomic
      state-change-plus-event commit.
- [ ] Choose the state boundary: one database for all app-owned stores, or
      named exceptions with reasons. Define the transaction API that a
      synchronous route and an asyncio callback both use safely without
      blocking the event loop.
- [ ] Define the migration policy for every legacy JSON store: idempotency,
      backup, validation, partial-migration recovery, rollback, downgrade, and
      corrupt-input diagnostics.
- [ ] Write `SPIKE.md` with the comparison and `DECISION.md` with the selected
      architecture, the rejected alternative and why, and the constraints the
      implementation tasks must honor.
- [ ] Re-check the three implementation tasks against the decision and refine
      their Steps where the chosen mechanism changes them.

## Definition of Done

- Both candidates are compared on every named axis, not asserted
  (cmd: `rg -n "SQLite|JSON snapshot|async|migration|backup|isolation" tasks/20260801-100405/SPIKE.md`).
- The design supports append-only events, correlation, idempotent delivery,
  retention, and atomic state-plus-event commits without creating the future
  conversation schema
  (cmd: `rg -n "append-only|correlation|idempotent|retention|outbox|atomic" tasks/20260801-100405/SPIKE.md`).
- A load-bearing choice with rejected alternatives is recorded
  (cmd: `test -f tasks/20260801-100405/DECISION.md && rg -n "Rejected|Alternative" tasks/20260801-100405/DECISION.md`).
- The migration and recovery policy covers every store the predecessor
  inventoried (cmd: `rg -n "idempotent|backup|rollback|partial|corrupt|downgrade" tasks/20260801-100405/DECISION.md`).
- The user accepts the durability and migration tradeoffs (manual: user check).

## Notes

- Epic: 20260729-102145.
- Depends on the inventory/reproduction spike.
- Decides the mechanism only. It migrates no production state.
- Do not pre-create speculative conversation tables; only prove the chosen
  store can carry them later through a normal schema migration.
