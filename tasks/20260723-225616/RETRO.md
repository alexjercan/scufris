# Retro: Read-only per-project skills+tools discovery + endpoint

- TASK: 20260723-225616
- BRANCH: feature/project-capabilities
- REVIEW ROUNDS: 1 (APPROVE, two no-change NITs)

## What went well

- Researched the codex/claude per-project conventions (paths + config schemas)
  BEFORE writing the discovery layer. The `_PROVIDER_SOURCES` registry matched
  the real conventions with zero rework, and the round-1 review was a clean
  APPROVE. Verifying dependency behavior before designing around it (the plan
  skill's rule) paid off directly.
- Test-first at the right altitude: the module tests drove the discovery
  functions and the endpoint test drove the HTTP contract; both are behavioral
  (would fail if the module were deleted), which the reviewer independently
  confirmed.
- Kept the diff honest and narrow: an unrelated pre-existing failure was NOT
  fixed inside this feature task (which would have widened the diff) but filed
  as its own task 20260723-233337.

## What went wrong

- The inherited red (`test_agent_config_omits_builtin_server_when_tools_disabled`)
  was discovered at VERIFY time, not at the start. Root cause: I did not run the
  full suite on the pristine base branch before implementing, so I had to spend a
  diagnosis pass (isolate the test on master, inspect `~/.local/state/scufris`)
  to prove the failure was not mine. The diagnosis was correct but the timing
  cost a detour - had I known the baseline was red from minute one, the failure
  would have been a non-event.

## What to improve next time

- Run the full check suite on the base branch commit BEFORE starting
  implementation, and note any pre-existing reds in TASK.md up front, so an
  inherited failure is known context rather than a verify-time surprise. (See
  the `check-the-base-suite-before-you-start` lesson.)

## Action items

- [x] Filed tatr 20260723-233337 (fix the test-isolation bug: the test reads the
  real state dir instead of an isolated tmp_path).
- [x] Ledger: added `check-the-base-suite-before-you-start` (x1).
