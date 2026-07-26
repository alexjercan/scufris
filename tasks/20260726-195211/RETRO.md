# Retro: isolate test_telegram lifespan tests from .env

- TASK: 20260726-195211
- BRANCH: fix/telegram-test-env-isolation
- REVIEW ROUNDS: 1 (APPROVE, in-session trivial diff)

## What went well

- The T5 flow Finish gate (full suite on the default branch, from the main
  checkout) caught a latent bug that every prior run had hidden: the sprout
  worktrees have no `.env`, so the suite was always green there. Running Finish
  from the `.env`-carrying main checkout is what exposed it.
- Diagnosis was fast and grounded: reproduced the exact failure on the T4
  pre-fix tree with the `.env` copied in, so the fix targeted the real cause
  (`.env` file leak) rather than a guess.

## What went wrong

- Pre-existing since T4: commit 444f627 fixed `.env` isolation in
  `test_config.py` but did not sweep the sibling `test_telegram.py` lifespan
  tests, which also build a real `Settings`. Root cause: the isolation fix was
  scoped to the file that visibly failed, not to the pattern ("any test building
  `Settings` for a defaults/explicit-input assertion").

## What to improve next time

- When fixing a test-isolation defect, grep the whole test tree for the same
  construction (`Settings(` here) and fix every instance in one pass, not just
  the file that happened to fail. New ledger entry
  `settings-test-must-disable-env-file` (x2).

## Action items

- [x] Ledger: added `settings-test-must-disable-env-file` (x2).
- No follow-up tasks: all `Settings(...)` sites in `test_telegram.py` now pass
  `_env_file=None`; `test_config.py` was already covered.
