# Spike: inventory app-owned mutable state and reproduce the write races

- STATUS: IN_PROGRESS
- PRIORITY: 85
- TAGS: spike, v0.2.0, reliability, storage
- KIND: SPIKE
- FLOW STEP: REVIEWING
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
- The read-modify-write windows are enumerated in their own section, each with
  the code location that opens it and what a concurrent writer costs
  (cmd: `rg -n "^### Read-modify-write windows" tasks/20260729-102146/SPIKE.md && rg -c "scheduler\.py:107|outcomes\.py:204|registry\.py:129|store\.py:456|reasoning_store\.py:82" tasks/20260729-102146/SPIKE.md`).
- The record lints clean (cmd: `tatr check 20260729-102146`).

## Notes

- Epic: 20260729-102145.
- Evidence only. The mechanism, migration, and recovery decision is the
  successor spike; splitting keeps the reproduction honest and the decision
  arguable against it.
- 20260729-124655 has landed, so the host proposal, approval, schedule, and
  digest stores are part of today's snapshot rather than a future one.

## Close-out

What and why. Round 1 returned REQUEST_CHANGES with three MAJOR findings, all
against the evidence rather than the prose. Addressing them added two things
the record was missing rather than restating what it had: the read-modify-write
enumeration Step 5 promised and never delivered, and the observation that a
failed persist leaves the record LIVE in memory. Both changed the
Recommendation, which is the part of this spike 20260801-100405 actually
consumes: it went from seven constraints to nine.

Alternatives. R1.5 could have been closed by removing the inverted exit code
instead of documenting it. Kept the inversion: reproducing a failure IS this
script's success condition, and a script that exits non-zero on a successful
reproduction is the more confusing artifact. Made it unmissable instead - a
distinct code 2 for the clean run, the warning first in the docstring, and the
code printed on both summary lines.

Difficulties and diagnosis. R1.1's own replacement number did not survive being
instrumented. The review claimed "45 of 110 finished agents lack an outcome",
measured by treating "create succeeded" as "finished". Splitting the counters
properly (`mark_finished_called` / `_raised` / `_returned`) showed
`returned_without_outcome: 0` - every call that returned cleanly landed all
three files, and the 45 came from calls that raised partway. The finding's
substance was right and its number was not tight enough. The figure now in the
record is the one the counters actually isolate: of 100 agents whose
`mark_finished` was called, all 100 got a session mapping and 35 ended with a
session and no outcome. Worth stating plainly - a review can be correct that a
number does not isolate its claim while offering a replacement that also does
not.

Evidence. All five DoD proofs pass, including the tightened R1.2 command that
the inventory table can no longer satisfy on its own. `tatr check` exits 0,
`ruff check` passes on the script, and the reproduction runs green (exit 0,
meaning reproduced) at commit 54714b7 on Linux x86_64 / 24 cores.

Reflection. The process signal from round 1 is the durable lesson: a `rg` proof
satisfiable by a section other than the one its Step is about is not a proof. It
let a genuinely undone step be ticked. The second-order lesson is from this
round - instrumenting to answer a review finding is worth doing even when the
finding already supplies a number, because the number in a finding is as
unverified as the one it replaces.
