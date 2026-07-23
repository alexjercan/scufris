# Retro: Fix test isolation - config tests read real state dir

- TASK: 20260723-233337
- BRANCH: fix/test-state-dir-isolation
- REVIEW ROUNDS: 1 (in-session, trivial diff; APPROVE, no findings)

## What went well

- Reproduce-first was free: the existing red WAS the reproduction, so the A/B
  proof (fails on the real populated state dir, passes after isolation with that
  dir untouched) fell straight out. This is exactly the payoff of the
  `check-the-base-suite-before-you-start` lesson that filed this task - the red
  was already characterized, so the fix cycle was pure execution.
- Scoped the sweep by MECHANISM (the override-managed `/api/agent/config` /
  `/api/agent/profiles` endpoints) rather than blindly isolating every
  `Settings(` in a 2900-line test file. That kept the diff to the two genuinely
  fragile tests and avoided churn on ~40 already-isolated constructions.
- Caught my own grep's false positives: an awk sweep for literal `state_dir`
  flagged tests that isolate via the `_mock_settings(tmp_path)` HELPER; verifying
  each before trusting the list avoided both a bogus finding and missing nothing.

## What went wrong

- Nothing in this cycle. The root cause itself (two tests defaulting `state_dir`
  to the real `~/.local/state/scufris`) is a latent fragility that only surfaced
  because a real dev override happened to disagree with a test's constructor
  arg - the kind of env-coupled test that passes on CI and a fresh machine but
  not on a working dev box.

## What to improve next time

- When a test asserts on state that a persisted override store can change, isolate
  `state_dir` at construction as a reflex - the store silently wins over the
  constructor arg. If this recurs, a conftest autouse fixture pointing `state_dir`
  at a per-test tmp for the whole suite kills the class outright (weighed against
  it: it would mask a test that legitimately wants a real dir).

## Action items

- [x] Ledger: added `isolate-state_dir-in-tests-that-assert-config` (x1), tagged
  as a conftest-autouse-fixture promotion candidate if it recurs.
- [x] Root-cause fix delivered: the pre-existing suite red is gone; future flow
  runs start from a green base.
