# Retro: Persisted agent<->session id registry

- TASK: 20260723-001251
- BRANCH: fix/session-registry (landed b877782)
- REVIEW ROUNDS: 1 (APPROVE, one MINOR fixed in the same round)

See TASK.md for what changed and DECISION.md for the mechanism choice; this is
process only.

## What went well

- Repro-first paid off cleanly: the restart reproduction was written before the
  fix and went red for the exact reported reason (`assert None == 'orch-sess'`),
  so the diagnosis was aimed, not guessed. The out-of-context reviewer
  re-derived the same red on master independently.
- Designing the registry to PRESERVE the whole observable `AgentStore` API meant
  zero existing tests changed - the diff is pure addition. That is a strong
  signal the refactor was behavior-preserving where it should be.
- The out-of-context review earned its keep: it found a real (if narrow) race
  (R1.1) that the in-session view had normalized as "pre-existing on master", and
  the registry's backend tag made it a cheap, well-pinned fix rather than a
  deferred follow-up.

## What went wrong

- Process slip: the first `SessionRegistry` edit landed in the MAIN checkout
  (`/home/alex/personal/scufris`), not the sprout worktree. Root cause: I Read
  `agent_store.py` at its main-checkout path DURING the plan phase (before
  sprouting), and the follow-up Edit reused that same path by reflex. Caught
  immediately by a `git status` on the main tree and reverted before any commit,
  so no harm - but it cost a redo.
- The API-preserving design has a flip side the review named: because no existing
  test changed, the OLD "session_id round-trips via agents.json" contract was
  never directly pinned - the new tests pin the new mechanism, but a reader can't
  see the old behavior was deliberately replaced rather than accidentally
  dropped. Acceptable here (the behavior is genuinely gone), but worth a beat of
  thought each time.

## What to improve next time

- After `sprout new`, re-Read a file from the WORKTREE path before the first Edit
  - never carry a path Read during planning (main checkout) into the work phase.
- On an API-preserving refactor, explicitly ask "what old contract am I dropping,
  and is anything still asserting it?" before trusting an all-green existing suite
  as proof of safety.

## Action items

- [x] Fixed R1.1 in-cycle (mark_finished keys by the run's backend); pinned by a
      sabotage-verified test. No follow-up task needed.
- Sibling bugs 20260721-152034 and 20260720-020345: disposition deferred to the
  flow Finish step (noted in TASK.md and GOAL.md), not silently dropped.
