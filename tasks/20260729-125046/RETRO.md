# Retro: Add scheduled host checks and a proactive digest

- TASK: 20260729-125046
- BRANCH: feat/host-digest
- REVIEW ROUNDS: 2 (5 findings: 2 MAJOR, 2 MINOR, 1 NIT)

What shipped is in `NOTES.md`. Process only here.

## What went well

- **The four forks went to the operator before any code.** "Code or a model turn",
  "what does a boring day sound like", "in-process or systemd", "is escalation on" -
  each one changed what got built, and settling them first meant the check set, the
  renderer and the scheduler could be written once.
- **Probing over TIME, not just over cases.** The two MAJORs both came from asking
  what happens on the fourth tick rather than the first. That is the question this
  feature lives or dies on, and it is not a question any single-pass test asks.
- **Running the example and reading it.** Two wording problems (the detail printed
  under the wrong check, and an unreadable check hiding inside an all-clear) were
  found by looking at output, before review.
- **The repo's guards did their job again**: the whitelist-sync test forced twelve
  settings into both lists, the route sweep exposed a blocking endpoint, and the
  strict tsc build caught an incomplete interface literal vitest accepted.

## What went wrong

- **I wrote a convenience nobody asked for, and it cost real host I/O in every
  test.** "Run once on a fresh schedule so the feature proves itself" was my
  invention. It made every app boot in the suite read the machine, doubled the suite's
  runtime, and would have hammered the host on a restart loop. Root cause: adding
  behaviour on the happy path without asking what it does on the paths that are not
  happy - a test boot, a crash loop.
- **Every test I wrote drove ONE pass.** The feature's whole risk is what it sounds
  like over a week, and nothing in the suite looked past a single tick until the
  review probed four. The DoD's own tests are all single-pass too, so the plan shares
  the blame - a proof list can be complete and still miss the dimension that matters.
- **The run-now endpoint blocked** because I reached for "await the thing" rather than
  asking how long the thing takes. The approve endpoint next door had already solved
  exactly this (start a supervised run, return, let the client poll) and I did not
  copy it.

## What to improve next time

- For anything that fires repeatedly, the FIRST test is "what does an unchanged world
  produce over N ticks". Single-pass tests cannot see the failure mode that gets a
  notification feature muted.
- Before adding a convenience to a startup path, write down what it does during a test
  boot and during a crash loop. If either answer is "real I/O", it does not belong
  there.
- When adding an endpoint that triggers work, look at how the nearest existing
  endpoint of the same shape handles duration before choosing to await.

## Action items

- [x] Lessons ledger: `notification-features-need-a-repetition-test` and
      `no-work-on-a-startup-path-a-test-boot-also-walks` appended.
- [x] `DECISION.md` sections 2 and 4 amended with what the build measured (the
      change-gate on `watch`, and the two escalation guards), so the record describes
      what was built.
- [ ] Not created as a task: the week of living with the digests (the manual
      acceptance), and an out-of-context review round.
