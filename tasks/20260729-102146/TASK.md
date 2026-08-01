# Spike: inventory app-owned mutable state and reproduce the write races

- STATUS: OPEN
- PRIORITY: 85
- TAGS: spike, v0.2.0, reliability, storage
- KIND: SPIKE
- FLOW STEP: PLANNED
- PLAN STATUS: APPROVED
- PARENT: 20260729-102145

## Story

As a maintainer, I want a verified inventory of every app-owned mutable store
plus a reproducible demonstration of its concurrency failures, so that the
persistence decision argues from measured evidence rather than from a
remembered picture of the code.

## Question

What mutable state does Scufris own today, who writes each store, and which of
those writers can collide? Answered with a measured reproduction, not a reading:
the successor spike (20260801-100405) has to argue its mechanism against this
evidence. Out of scope here: choosing the mechanism, the migration, or the
recovery policy.

## Steps

- [x] Inventory every mutable state store that exists today, one row per store:
      module, on-disk path, write pattern, record shape, and who mutates it.
      Cover `scufris/projects.py`, `scufris/settings_store.py`,
      `scufris/reasoning_store.py`, `scufris/digest.py`, `scufris/scheduler.py`,
      `scufris/auth/store.py`, `scufris/agent_store/{store,registry,outcomes}.py`,
      and `scufris/host_approvals.py`.
- [x] Record the root-owned `scufris/hostd/audit.py` log as an intentional
      external boundary, with the reason it stays outside the app store.
- [x] For each store, classify the mutators: synchronous FastAPI thread-pool
      routes, supervisor/asyncio callbacks, Telegram handlers, scheduler ticks.
      Name the pairs that can write the same file at the same time.
- [x] Reproduce the fixed shared temporary-file race with a runnable script
      under `tasks/<id>/`; record the observed failure (traceback, lost record,
      or truncated file) and the concurrency needed to trigger it.
- [x] Enumerate the remaining lost-update and partial-write windows found by
      read-modify-write inspection, each with the code location that opens it.
- [x] Write `SPIKE.md` with the inventory table, the mutator matrix, and the
      reproduction evidence. No mechanism choice here.

## Definition of Done

- The inventory names every store module and the external audit boundary
  (cmd: `rg -n "projects|settings_store|reasoning_store|digest|scheduler|auth/store|agent_store|host_approvals|hostd/audit" tasks/20260729-102146/SPIKE.md`).
- Each store row records its writers and whether they can overlap
  (cmd: `rg -n "thread-pool|supervisor|scheduler|telegram|overlap" tasks/20260729-102146/SPIKE.md`).
- The race is reproduced by a committed script, not described from memory
  (cmd: `ls tasks/20260729-102146/repro_*.py && rg -n "observed|traceback|lost" tasks/20260729-102146/SPIKE.md`).
- The lost-update windows are listed with file and line references
  (cmd: `rg -n "scufris/.*\.py:[0-9]+" tasks/20260729-102146/SPIKE.md`).
- The record lints clean (cmd: `tatr check 20260729-102146`).

## Notes

- Epic: 20260729-102145.
- Evidence only. The mechanism, migration, and recovery decision is the
  successor spike; splitting keeps the reproduction honest and the decision
  arguable against it.
- 20260729-124655 has landed, so the host proposal, approval, schedule, and
  digest stores are part of today's snapshot rather than a future one.
