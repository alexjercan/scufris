# Retro: app-server turn timeout as an idle guard, not a wall-clock cap

- TASK: 20260724-011406
- BRANCH: bug/agent-idle-timeout
- REVIEW ROUNDS: 1 (APPROVE)

## What went well

- Diagnosis before code paid off: reading `supervisor.py` alongside `agent.py`
  surfaced that ADR-001 had already moved to a stall guard and the runner
  deadline was a leftover contradiction. That reframed the fix from "raise the
  120s" to "the runner has the wrong timeout MODEL", which is what the user
  actually wanted.
- Reproduce-first with the existing fake-app-server harness gave a real,
  sub-second red test (`app-server new timed out`) before any edit, so the fix
  was aimed, not guessed. The idle bound (0.4s) sat between the delta gap
  (0.15s) and the total (0.75s), pinning the fix at its own boundary.
- Surfacing the two design forks (idle-vs-remove, retry scope) at the gate
  BEFORE planning meant the plan matched intent on the first pass - no rework.
- The fix was small because the existing `except (TimeoutError,
  asyncio.TimeoutError)` already did the right thing; letting `wait_for` raise
  instead of adding a second timeout branch kept one error path, not two.

## What went wrong

- Nothing broke, but two doc-accuracy nits escaped implementation (R1.1, R1.2).
  Root cause: I rewrote the `agent_timeout_seconds` docstring thinking only
  about the codex runner I was editing, forgetting the same setting is read by
  the opencode backend; and I did not sweep the ledger for entries the fix
  invalidated. Both are the same blind spot - a shared symbol / a lesson has
  readers beyond the file in front of me.
- Minor friction: bare `pytest` in the sprout tested the main checkout
  (conftest guard) - a known lesson I re-hit rather than recalled up front.

## What to improve next time

- When editing the docstring/semantics of a SHARED config field, grep its
  readers (`grep -rn <field> scufris/`) before rewording, so the doc stays true
  for every consumer, not just the one being changed.
- When a fix invalidates an existing pattern, grep LESSONS.md for that pattern
  in the same cycle (a mid-flow lesson applies backward) - the reviewer caught
  the stale wall-clock lesson that I should have swept.

## Action items

- [x] R1.1/R1.2 addressed on-branch this cycle (docstring note + LESSONS fix).
- No new follow-up tasks: the opencode/mcp alignment (20260724-081804) and the
  retry spike (20260724-081811) were already filed at plan time.
