# Spike: choose the transactional state persistence architecture

- STATUS: OPEN
- PRIORITY: 0
- TAGS: spike,backlog,reliability,storage

## Story

As a maintainer, I want one durable persistence design for all mutable Scufris
state, so that concurrent agents, request threads, process crashes, and schema
changes have explicit correctness guarantees rather than store-specific JSON
behavior.

## Steps

- [ ] Inventory every mutable state store, its schema, write paths, ownership,
      recovery behavior, and compatibility requirements.
- [ ] Reproduce and quantify the fixed-temporary-file race and identify other
      lost-update or partial-write windows.
- [ ] Compare SQLite transactions with locked atomic JSON snapshots, including
      async/thread interaction, migrations, backups, observability, and test
      isolation.
- [ ] Choose the state boundary and transaction model, including whether all
      stores share one database and how multi-record updates are committed.
- [ ] Define idempotent migration, rollback, backup, corruption recovery, and
      downgrade behavior for existing JSON state.
- [ ] Write `SPIKE.md` with the evidence and `DECISION.md` with the selected
      architecture, rejected alternatives, and constraints.
- [ ] Refine child task 20260729-102147 if the chosen architecture changes its
      implementation or verification plan.

## Definition of Done

- The spike accounts for projects, agents, sessions, outcomes, settings,
  reasoning, and future plugin state (cmd: `rg -n "projects|agents|sessions|outcomes|settings|reasoning|plugin" tasks/20260729-102146/SPIKE.md`).
- A load-bearing persistence choice and migration policy are recorded
  (cmd: `test -f tasks/20260729-102146/SPIKE.md && test -f tasks/20260729-102146/DECISION.md && tatr check 20260729-102146`).
- The original concurrent-create failure is captured as a reproducible test
  target for the implementation task
  (cmd: `rg -n "concurrent|restart|migration" tasks/20260729-102146/SPIKE.md`).
- manual: the user accepts the selected durability and migration tradeoffs.

## Notes

- Epic: 20260729-102145.
- Relevant code: `scufris/projects.py`, `scufris/agent_store.py`,
  `scufris/settings_store.py`, and `scufris/reasoning_store.py`.
- This task decides the mechanism. It does not migrate production state.

## Flow State

- FLOW STEP: PLANNING
