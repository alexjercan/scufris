# Retro: Q1-A carry in-flight prompt on run status + inject user bubble on codex reattach

- TASK: 20260724-141430
- BRANCH: fix/reattach-prompt
- REVIEW ROUNDS: 1 (APPROVE, out-of-context, zero findings)

See TASK.md close-out for what/why and REVIEW.md for the round. Process only here.

## What went well

- Grounded the plan in code reads before writing: re-read RunState/_Run/start,
  the status endpoint, _launch_agent_turn, and the runTurn/reattach/settle
  interaction, plus a real codex rollout on disk. The design (status endpoint as
  the prompt carrier, deferred ensureBubble ordering) was right first try - the
  APPROVE with zero findings reflects that, not luck.
- Test-first with revert-sensitivity checks: after green, independently neutered
  the injection and the dedup guard and confirmed each new test fails at its OWN
  boundary. This pre-empted the reviewer's "would it fail if reverted?" and the
  no-duplicate test's vacuous-pass risk (it passes trivially unless the guard is
  what's under test).
- Kept the diff focused: `ruff format` reflowed three unrelated files (pre-existing
  format drift on master); reverted them so the task diff is only the 8 intended
  files.

## What went wrong

- The plan's step 3 asserted the captured prompt "still carries the steering
  preamble (added inside the backend per turn)". It does not: `_steer` runs
  DOWNSTREAM at agent.py:583 inside the codex turn path, so the prompt captured at
  _launch_agent_turn is already raw/unsteered. Root cause: the plan located the
  steering transform from a mental model of the architecture instead of grepping
  the `_steer` call site. Caught during work (reading agent.py before wiring the
  strip) and the step text was corrected, so it cost only the correction - but it
  is exactly the class of plan error that has bitten before when not caught.

## What to improve next time

- When a plan step asserts WHERE a transform happens (before/after a call, which
  layer), grep the call site and cite it in the step, or phrase it verify-first -
  do not encode a location from the architecture model. (This is already the plan
  skill's rule; the miss here is a reminder it applies to "where", not just "what".)

## Action items

- [x] Ledger: added `plan-locates-transform-from-the-call-site-not-the-model`
      (x1, -> plan skill).
- No follow-up code work. Q2 remains parked as task 20260724-141150 (p0).
