# Retro: Migrate agent, session, outcome, settings, and reasoning state

- TASK: 20260801-100409
- BRANCH: fix/migrate-agent-session-outcome-settings-reasoning-state
- REVIEW ROUNDS: 2

## What went well

- Landing the `transaction()` loop-thread guard BEFORE the store rewrites made
  the caller sweep failure-driven rather than reading-driven. It kept paying
  after the fact: a round-2 sabotage that moved a store call back onto the loop
  was caught by the guard before the responsiveness proof even ran.
- Per-source legacy gate rows plus one fixture directory per shape meant R1.5
  was a ten-line change to the loop, not a redesign.
- Making the R1.1 proof deterministic by slowing `mark_running` 50ms in the test
  (3/5 red -> 5/5 red) without changing what it asserts. The defect is that the
  window exists; the test should not have to win a race to say so.

## What went wrong

- DECISION.md 2 turned `_launch_agent_turn` and `persist` into coroutines, and
  three separate check-then-act sequences that were atomic by construction
  silently became interleavable: the one-run-per-agent guard (R1.1, BLOCKER),
  `fork_session`'s clear-then-launch (R1.3), and the wake bridge's drain (R1.4).
  Each of the three carried a comment asserting the atomicity it had just lost,
  so the diff shipped with three false load-bearing comments.
  Why it seemed sound: the decision was reviewed for ORDER - "the loop-bound
  tail stays on the loop, in its current order" - and order was in fact
  preserved. The question never asked was which sequences depended on there
  being no suspension point at all.
- The DoD's responsiveness proof (R1.2) could not fail on its stated criterion:
  the lock was released before the routes ran, and the tick assertion was
  satisfied by a preceding sleep. A proof written alongside the fix inherits the
  author's belief that the fix works.
- Two unrelated files were reformatted (R1.6) by the same bulk test sweep the
  Steps had explicitly warned about after 20260801-120412.

## What to improve next time

- Breadth: 3695 insertions over 54 files. The agent/session/outcome trio is
  genuinely inseparable - the Story's guarantee IS their single commit - but
  `settings_store` and `reasoning_store` share no transaction with it and were
  independently landable behind the same guard. Two smaller branches would have
  put the completion-path concurrency change in a diff a reviewer could hold.
- Churn: the plan-time question that would have caught R1.1/R1.3/R1.4 is not the
  from-scratch challenge but a consequence sweep on the decision itself - when a
  decision converts a synchronous call path to an awaited one, enumerate the
  check-then-act sequences it splits. The grep is cheap and mechanical: every
  comment saying "synchronous", "atomic" or "cannot interleave" adjacent to a
  new `await` in the diff.
- Write the liveness/responsiveness proof against the stimulus it names, then
  sabotage it deliberately. "It passes" is not evidence for a proof whose whole
  job is to fail on one condition.

## Action items

- Carried into the successor task (auth, host, schedule, digest state): run the
  await-splits-a-check-then-act sweep as an explicit Step, since that task
  inherits the same coroutine `on_complete` contract.
- R2.1 (MINOR, accepted open): `fork_session`'s rewritten comment still denies an
  interleaving the `asyncio.to_thread` resume boundary permits. Harm is a 409 on
  the operator's fork, not a lost seed. Fold into the successor task's sweep
  rather than reopening this branch.
