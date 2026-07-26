# Review: isolate test_telegram lifespan tests from .env

- TASK: 20260726-195211
- BRANCH: fix/telegram-test-env-isolation

## Round 1

- VERDICT: APPROVE
- REVIEWER: in-session (trivial diff: test-only, adds `_env_file=None` to four
  `Settings(...)` calls, mirroring the established `tests/test_config.py` pattern
  from T4 commit 444f627 - no behavior change, no production code touched)

Verified the fix reproduces and resolves the failure: with the real `.env`
(carrying `SCUFRIS_TELEGRAM_BOT_TOKEN`) copied into the worktree, the T4 pre-fix
version of `test_no_bot_without_token` + `test_on_reset_clears_session_serialized`
failed (2 failed) and the fixed version passed (29 passed); the full suite ran
green from that `.env`-containing tree (ruff clean, mypy "no issues found in 52
source files", pytest exit 0). No findings.

No `manual:` DoD items.
