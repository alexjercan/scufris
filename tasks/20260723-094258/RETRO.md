# Retro: BC1 durable run-outcome record + AgentState.WAITING

- TASK: 20260723-094258
- BRANCH: feat/run-outcome-record
- REVIEW ROUNDS: 2

## What went well

- Mirroring the just-landed `SessionRegistry` sidecar (atomic write, tolerant
  load, corrupt-file test) made the store fast and low-risk to build - the
  pattern is well-worn, and reusing it kept the diff small and conventional.
- Test-first red->green worked cleanly for the substrate: all four happy-path
  tests were watched failing for the RIGHT reason (missing enum member, missing
  kwarg, missing accessor) before the code existed.
- The out-of-context reviewer earned its keep: it found a real MAJOR the
  happy-path tests missed, and I re-verified it in-session (wrote the regression
  test, watched it resurrect a stale outcome) before trusting the finding.

## What went wrong

- R1.1 (MAJOR): I placed the new outcome write BEFORE the regular-agent `_raw`
  existence check in `mark_finished`, so an agent deleted mid-run had a stale
  outcome resurrected that survived restart. Root cause: I copied
  `SessionRegistry`'s SHAPE (the sidecar class) but reasoned about the write
  PLACEMENT independently - "write once for all agents, above the split, is
  uniform/cleaner" - instead of mirroring where the sibling `SessionRegistry.set`
  already sits (AFTER `_raw`, precisely so a deleted agent never leaks). The
  existence check is load-bearing for the delete guarantee, and the delete-mid-run
  path was even documented in the `app.py` persist comment I had read.

## What to improve next time

- When mirroring an existing store/pattern, copy its GUARD ORDERING too, not just
  the class skeleton: where a write sits relative to the existence/validation
  check is part of the pattern.
- Any NEW persisted write added to a run-completion / on-complete callback keyed
  by an entity id needs a delete-mid-run regression test - the callback can fire
  after the entity is gone (the code often already says so).

## Action items

- [x] Ledger: added `completion-callback-write-after-existence-check` (x1).
- No follow-up code tasks: BC2-BC5 already seeded and depend on this substrate.
</content>
