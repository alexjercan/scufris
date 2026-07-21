# Retro: B5bc - retire the Agent protocol + move orchestrator sessions

- TASK: 20260721-180208
- BRANCH: feature/unify-orchestrator
- REVIEW ROUNDS: 1 (APPROVE, out-of-context, 1 MINOR + 2 NITs, all addressed)

## What went well

- Reading the code before touching it paid the whole cost back. The reroute
  looked like ~10 endpoints + 5 classes + ~50 test touchpoints, but the per-agent
  machinery (`_launch_agent_turn`, `_relay_bus_sse`, the reserved orchestrator
  record from B5a) already existed, so the landing endpoints became thin
  delegations rather than new code. The orchestrator was already a first-class
  agent via `/api/agents/orchestrator/chat`; the landing endpoints just had to
  point at it.
- The net diff is -354 lines. Retiring the abstraction genuinely simplified the
  system instead of moving complexity around - the clearest signal the merge of
  B5b+B5c into one all-or-nothing cut was the right call.
- Bisecting the hang instead of guessing. When pytest stalled at ~30%, a per-test
  `timeout 25` loop over the rewritten tests pinned it to
  `test_fork_seeds_new_session_with_prior_context` in one run, which made the
  root cause (nested serialize-key acquire) obvious. Guessing would have wasted
  far longer.
- Out-of-context review earned its keep: it verified the deadlock fix against the
  ACTUAL supervisor FIFO-lock implementation (not my description of it) and
  caught that the cross-backend-clear behavior had lost its only test guard when
  the AgentHandle test was deleted.

## What went wrong

- fork self-deadlocked (BLOCKER, caught by tests before review). Root cause: I
  rerouted fork by keeping its old `async with supervisor.serialized("chat")`
  wrapper and swapping the body to `_launch_agent_turn`, which itself reserves
  the same serialize key inside `supervisor.start`. The old body (`agent.chat`)
  never touched the supervisor, so the wrapper was safe; the new body does, so
  the wrapper became a nested acquire of a non-reentrant per-key lock. I changed
  what the lock body does without re-checking whether the lock was still safe to
  hold. Lesson `serialize-then-launch-self-deadlocks-on-shared-key`.
- Doc-sweep miss: the `agent_runs` comment still described the retired `"chat"`
  serialize key after I moved everything to `ORCHESTRATOR_ID`. The review's NIT
  caught it. The work-skill doc sweep should have grepped for `"chat"` (the
  string I was removing) across comments, not just code.
- The deleted-test gap: retiring `AgentHandle` deleted
  `test_agent_handle_rebuilds_and_carries_session`, and the REPLACEMENT behavior
  (clear-on-switch, the inverse) had no app-level test until the review flagged
  it. When a retirement inverts a behavior, the new behavior needs its own guard
  landed in the same task, not just the old test removed.

## What to improve next time

- When a change moves code INTO or OUT of a held lock's body (or swaps what the
  body calls), re-derive the lock safety from scratch - re-entrancy, key
  identity, what the new body acquires. A lock that was safe around the old body
  is not automatically safe around the new one.
- Doc sweep on a rename/removal greps for the OLD token (here `"chat"`) across
  comments and docstrings, not only symbols in code.
- When retiring a class deletes a behavioral test, ask "what replaced that
  behavior, and does the replacement have its own falsifiable test?" before
  closing - the inverse behavior is the easiest guarantee to silently drop.

## Action items

- [x] Lesson `serialize-then-launch-self-deadlocks-on-shared-key` added to
      LESSONS.md (domain).
- Manual acceptance (batched to Finish): hold a multi-turn landing/orchestrator
  conversation, switch sessions, confirm end-to-end.
