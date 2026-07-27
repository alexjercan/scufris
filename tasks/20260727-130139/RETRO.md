# Retro: Test isolation - _ensure_den_path leaks SCUFRIS_DEN_PATH across tests

- TASK: 20260727-130139
- BRANCH: fix/den-env-test-isolation
- REVIEW ROUNDS: 1 (APPROVE)

## What went well

- Reading the ledger first paid off literally: the lesson
  `os-environ-setdefault-in-test-leaks-past-monkeypatch` already diagnosed this
  class and even named the fix ("a conftest autouse fixture that
  snapshots/restores `SCUFRIS_*` keys, not per test") and this task id. Zero
  design search - the ledger handed over the approach. This is the compounding
  working as intended.
- Reproduced the exact failure before touching code (test_app then the hermetic
  test_backends test, red), so the fix was aimed, not guessed.
- Extracting `snapshot_scufris_env` / `restore_scufris_env` as named helpers let
  the regression test pin the mechanism DETERMINISTICALLY instead of relying on
  inter-test ordering (fragile under pytest-randomly). Clean one-round APPROVE.

## What went wrong

- Nothing structural. Two NITs from review: a dead `_ensure_den_path` call in
  the pre-existing-key test (setdefault is a no-op when the key exists, so the
  call proved nothing), and a missing note that the snapshot deliberately
  includes the just-set `SCUFRIS_STATE_DIR`. Root cause of the dead call: the
  second test was written by analogy to the first without re-checking that
  `setdefault` short-circuits on a present key. Both addressed in round 1.

## What to improve next time

- When a bug only manifests with an ambient `.env` and the work happens in a
  sprout worktree (which has none), reproduce it by writing a temporary `.env`
  into the worktree, verify, then delete it. Relying on "green in the worktree"
  would have hidden whether the fix actually closed the leak. Worth a lesson.

## Action items

- [x] Ledger: bump `os-environ-setdefault-in-test-leaks-past-monkeypatch` and
      mark it RESOLVED - the conftest snapshot/restore guard it prescribed now
      exists.
- [x] Ledger: file `env-dependent-bug-repro-needs-a-temp-dotenv-in-the-worktree`
      (x1) - a leak that only fires with an ambient `.env` must be reproduced by
      seeding a temp `.env` in the worktree, since sprout worktrees have none.
- No follow-up code work: the fix is general (all `SCUFRIS_*` keys), not
  point-patched to `SCUFRIS_DEN_PATH`.
